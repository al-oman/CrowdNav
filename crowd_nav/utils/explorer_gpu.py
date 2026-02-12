import logging
import copy
import torch
from crowd_sim.envs.utils.info import *

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Single transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, states, values):
        """
        Push batched states and values
        states: tensor [batch, state_dim]
        values: tensor [batch]
        """
        for s, v in zip(states, values):
            self.push((s, v))

    def sample(self, batch_size):
        import random
        batch = random.sample(self.memory, batch_size)
        states, values = zip(*batch)
        states = torch.stack(states)
        values = torch.stack(values)
        return states, values

    def __len__(self):
        return len(self.memory)


class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                    print_failure=False, num_envs=16):
        """
        Run k episodes in parallel using num_envs vectorized environments.
        Observations are batched and moved to GPU for policy forward pass.
        """
        self.robot.policy.set_phase(phase)

        # initialize parallel environments
        envs = [copy.deepcopy(self.env) for _ in range(num_envs)]
        dones = [False] * num_envs
        episode_counts = [0] * num_envs
        cumulative_rewards = [0.0] * num_envs
        states_batch = [[] for _ in range(num_envs)]
        actions_batch = [[] for _ in range(num_envs)]
        rewards_batch = [[] for _ in range(num_envs)]

        # statistics
        success_times, collision_times, timeout_times = [], [], []
        success, collision, timeout = 0, 0, 0
        too_close, min_dist = 0, []

        total_episodes = 0
        while total_episodes < k:
            obs_list = []
            # reset environments that are done
            for i, env in enumerate(envs):
                if dones[i]:
                    obs = env.reset(phase)
                    obs_list.append(obs)
                    dones[i] = False
                else:
                    obs_list.append(env._last_obs)  # keep last obs
            obs_tensor = torch.stack([torch.tensor(obs, dtype=torch.float32)
                                    for obs in obs_list]).to(self.device)  # [num_envs, obs_dim]

            # policy forward pass for all envs
            with torch.no_grad():
                actions_tensor = self.robot.act(obs_tensor)  # must support batched input
            actions_np = actions_tensor.cpu().numpy()

            # step all envs
            infos = []
            next_obs_list = []
            rewards = []
            for i, env in enumerate(envs):
                if not dones[i]:
                    next_obs, reward, done, info = env.step(actions_np[i])
                    next_obs_list.append(next_obs)
                    rewards.append(reward)
                    dones[i] = done
                    infos.append(info)

                    # store states/actions/rewards
                    states_batch[i].append(self.robot.policy.last_state)
                    actions_batch[i].append(actions_np[i])
                    rewards_batch[i].append(reward)

                    if isinstance(info, Danger):
                        too_close += 1
                        min_dist.append(info.min_dist)
                else:
                    # placeholder for done envs
                    next_obs_list.append(obs_list[i])
                    rewards.append(0.0)
                    infos.append(None)

            # update cumulative rewards
            for i in range(num_envs):
                cumulative_rewards[i] += rewards[i]

            # check for finished episodes and update memory/stats
            for i, done_flag in enumerate(dones):
                if done_flag:
                    total_episodes += 1
                    info = infos[i]
                    if isinstance(info, ReachGoal):
                        success += 1
                        success_times.append(envs[i].global_time)
                    elif isinstance(info, Collision):
                        collision += 1
                        collision_times.append(envs[i].global_time)
                    elif isinstance(info, Timeout):
                        timeout += 1
                        timeout_times.append(envs[i].time_limit)
                    else:
                        raise ValueError('Invalid end signal from environment')

                    if update_memory and (isinstance(info, ReachGoal) or isinstance(info, Collision)):
                        self.update_memory(states_batch[i], actions_batch[i], rewards_batch[i], imitation_learning)

                    # reset batch lists for this env
                    states_batch[i], actions_batch[i], rewards_batch[i] = [], [], []

            # update observations for next step
            for i in range(num_envs):
                envs[i]._last_obs = next_obs_list[i]

        # log statistics
        success_rate = success / k
        collision_rate = collision / k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else envs[0].time_limit

        extra_info = '' if episode is None else f'in episode {episode} '
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                    format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            sum(cumulative_rewards) / k))

        num_steps = sum(success_times + collision_times + timeout_times) / self.robot.time_step
        logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                    too_close / num_steps, sum(min_dist) / len(min_dist) if min_dist else 0)

        def update_memory(self, states, actions, rewards, imitation_learning=False):
            """
            Vectorized update_memory for batched states from multiple environments.
            Assumes states is a list of tensors of shape [state_dim] or [human_num, feature_dim].
            """
            if self.memory is None or self.gamma is None:
                raise ValueError('Memory or gamma value is not set!')

            # Stack states into one tensor: [batch, state_dim]
            states_tensor = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
                                        for s in states]).to(self.device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

            if imitation_learning:
                # IL: compute cumulative discounted rewards for each state
                batch_size = len(states)
                values = torch.zeros(batch_size, device=self.device)
                for i in range(batch_size):
                    values[i] = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * rewards[t]
                                    for t in range(batch_size)])
            else:
                # RL: compute target value using target model for non-terminal states
                with torch.no_grad():
                    next_states = states_tensor[1:]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    values = torch.zeros(len(states_tensor), device=self.device)
                    for i in range(len(states_tensor)):
                        if i == len(states_tensor) - 1:
                            values[i] = rewards_tensor[i]
                        else:
                            values[i] = rewards_tensor[i] + gamma_bar * self.target_model(next_states[i].unsqueeze(0)).item()

            # push batched states and values to memory
            self.memory.push_batch(states_tensor, values)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
