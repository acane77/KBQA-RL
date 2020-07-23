import torch
from state import State
import torch.nn.functional as F

class Reward:
    def __init__(self):
        pass

    def __call__(self, next_state:State, answer: str, **kwargs):
        # NOTE: return type is a pytorch tensor
        raise NotImplementedError

class ToyRewardFunc(Reward):
    def __init__(self):
        super().__init__()

    def __call__(self, next_state:State, answer: str, **kwargs):
        if next_state.e_t == answer:
            return torch.tensor(100.)
        else:
            return torch.tensor(0.)

class CosineSimiliarityReward(Reward):
    reward_dict = {}

    def __init__(self):
        super().__init__()
        self.discount_r = 0.8

    def __call__(self, next_state: State, answer: str, **kwargs):
        t = next_state.t
        current_state = kwargs.get('current_state')
        action = kwargs.get('action')
        SAS = (current_state, action, next_state)
        _R = CosineSimiliarityReward.reward_dict.get(SAS)
        if (_R): return _R
        F_phi = self.discount_r * self.phi(next_state) - self.phi(current_state)
        R = 1 if next_state.e_t == answer else 0
        reward = R + F_phi
        CosineSimiliarityReward.reward_dict[SAS] = reward
        return reward

    def phi(self, state):
        t = state.t
        if t == 0:
            return torch.tensor(0.)
        H_t = state.H_t[t-1]
        Q_t = torch.stack(state.q_t).sum(dim=(0, 1))  ## 将问题q_t的的每一维相加
        reward = torch.cosine_similarity(Q_t, H_t, dim=0)
        reward = reward.sum()
        return torch.relu(reward)