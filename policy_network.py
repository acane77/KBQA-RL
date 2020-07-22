import torch
import torch.nn as nn
from nets.attention import Attention
from nets.perceptron import Perceptron
from expeiment_settings import ExpSet
from nets.lstm import GRU
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.use_attention = ExpSet.use_attention
        self.use_perceptron = ExpSet.use_perceptron

        self.attention = Attention(ExpSet.dim)
        self.perceptron4hq = Perceptron(2 * ExpSet.dim, ExpSet.dim, ExpSet.dim)
        self.slp = nn.Linear(in_features=ExpSet.dim, out_features=ExpSet.dim)
        self.gru = GRU(ExpSet.dim, ExpSet.dim // 2, 2, ExpSet.dim)
        self.perceptron = Perceptron(ExpSet.dim, ExpSet.dim, 2 * ExpSet.dim)

    def forward(self, action_embedding: torch.Tensor, question_t: torch.Tensor, history_t: torch.Tensor):
        '''
        :param action_embedding: 动作空间的嵌入 [ m x d ]
        :param question_t: 问题向量 [ n x d ]
        :param history_t: 编码的历史记录 [ d ]
        :return: 动作分布 [ d ]
        '''
        m = action_embedding.shape[0]
        # Attention Layer: Generate Similarity Scores between q and r and current point of attention
        if self.use_attention:
            question, _ = self.attention(question_t.expand([m, *question_t.shape]), action_embedding.reshape(m, 1, -1))
        else:
            question = question_t.expand([m, *question_t.shape])
        question = question.sum(dim=1)
        # Perceptron Module: Generate Semantic Score for action given q
        if self.use_perceptron:
            question_with_history = self.perceptron4hq(torch.cat([history_t.expand(question.shape), question], dim=1))
            semantic_scores = (question_with_history * action_embedding).sum(dim=1)
        else:
            action_embedding = F.normalize(action_embedding, p=2)
            question = F.normalize(question, p=2)
            semantic_scores = (action_embedding * question)  ## TODO: check it!! l2_normlize
        action_distribution = torch.softmax(semantic_scores, dim=0)
        return action_distribution
