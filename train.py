#!/usr/bin/python3

from dataset import Dataset
from policy_network import PolicyNet
from reinforcement_learning import ReinforcementLearning
from expeiment_settings import ExperimentSettings
import numpy as np
import torch

def main():
    ExperimentSettings.parse_args()
    torch.manual_seed(ExperimentSettings.seed)
    if ExperimentSettings.enable_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(ExperimentSettings.seed)
    dataset = Dataset(ExperimentSettings.path_KB, ExperimentSettings.path_QA, 0.8, ExperimentSettings.enable_cache)
    policy_net = PolicyNet(ExperimentSettings.use_attention, ExperimentSettings.use_perceptron)
    rl = ReinforcementLearning(dataset=dataset, policy_net=policy_net,
                               num_epochs=ExperimentSettings.epochs, num_episode=ExperimentSettings.episodes,
                               steps=ExperimentSettings.max_T)
    rl.train()
    rl.test()
    rl.save_model('policy_net_{}'.format(np.random.randint(1e6, 1e7)))

if __name__ == '__main__':
    main()