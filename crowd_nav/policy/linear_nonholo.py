import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot


class LinearNonholo(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = 'unicycle'
        self.multiagent_training = True

    def configure(self, config):
        assert True

    def predict(self, state):
        self_state = state.self_state
        desired_theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
        r = (desired_theta - self_state.theta + np.pi) % (2*np.pi) - np.pi 
        v = self_state.v_pref
        action = ActionRot(v, r)

        return action
