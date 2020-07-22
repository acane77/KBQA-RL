import torch
from state import State
import torch.nn.functional as F

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
        if t == 0:
            return torch.tensor(0.)
        H_t = next_state.H_t[t-1]
        Q_t = torch.stack(next_state.q_t).sum(dim=(0, 1))  ## 将问题q_t的的每一维相加
        Q_t = F.normalize(Q_t, dim=0, p=2)   ## TODO: FIX NORM
        H_t = F.normalize(H_t, dim=0, p=2)   ## TODO: FIX NORM
        reward = torch.cosine_similarity(Q_t, H_t, dim=0)
        if answer == next_state.e_t:
            reward = reward + torch.tensor(1)
        return reward
