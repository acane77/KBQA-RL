#!/usr/bin/python3

from dataset import Dataset
from policy_network import PolicyNet
from reinforcement_learning import ReinforcementLearning
from expeiment_settings import ExpSet
import numpy as np
import torch

def main():
    ExpSet.parse_args()
    torch.manual_seed(ExpSet.seed)
    if ExpSet.enable_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(ExpSet.seed)
    dataset = Dataset(ExpSet.path_KB, ExpSet.path_QA, 0.8, ExpSet.enable_cache)
    policy_net = PolicyNet(ExpSet.use_attention, ExpSet.use_perceptron)
    rl = ReinforcementLearning(dataset=dataset, policy_net=policy_net,
                               num_epochs=ExpSet.epochs, num_episode=ExpSet.episodes,
                               steps=ExpSet.max_T)
    rl.train()
    rl.test()
    rl.save_model('results/policy_net_{}'.format(np.random.randint(1e6, 1e7)))

if __name__ == '__main__':
    main()