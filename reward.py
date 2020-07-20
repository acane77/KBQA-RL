import torch
from state import *

class Reward:
    def __init__(self):
        pass

    def __call__(self, next_state:State, answer: str):
        # NOTE: return type is a pytorch tensor
        raise NotImplementedError

class ToyRewardFunc(Reward):
    def __init__(self):
        super().__init__()

    def __call__(self, next_state:State, answer: str):
        if next_state.e_t == answer:
            return torch.tensor(100)
        else:
            return torch.tensor(0)

class CosineSimiliarityReward(Reward):
    def __init__(self):
        super().__init__()

    def __call__(self, next_state: State, answer: str):
        t = next_state.t
        if t == 1:
            return 0
        H_t = next_state.H_t[t]
        Q_t = next_state.q_t.sum(dim=(0, 1))  ## 将问题q_t的的每一维相加
        Q_t = torch.norm(Q_t)
        H_t = torch.norm(H_t)
        reward = torch.cosine_similarity(Q_t, H_t, dim=0)
        if answer == next_state.e_t:
            reward = reward + torch.tensor(1)
        return reward