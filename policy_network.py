import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.attention import Attention
from nets.perceptron import Perceptron
from expeiment_settings import ExpSet
from nets.lstm import GRU

class PolicyNet(nn.Module):
    def __init__(self, use_attention, use_perceptron):
        super(PolicyNet, self).__init__()

        self.use_attention = use_attention
        self.use_perceptron = use_perceptron
        self.embedder = None  #initialized in rl class

        self.attention = Attention(ExpSet.dim)
        self.perceptron4hq = Perceptron(2 * ExpSet.dim, ExpSet.dim, ExpSet.dim)
        self.slp = nn.Linear(in_features=ExpSet.dim, out_features=ExpSet.dim)
        self.gru = GRU(ExpSet.dim, ExpSet.dim // 2, 2, ExpSet.dim)
        self.perceptron = Perceptron(ExpSet.dim, ExpSet.dim, 2 * ExpSet.dim)

    def forward(self, action_space, question_t: torch.Tensor, history_t: torch.Tensor):
        semantic_scores = []
        actions = []
        for action in action_space:
            ## TODO: 提前load适应GPU
            rel = self.embedder.get_relation_embedding(action)
            if rel is not None:
                # Attention Layer: Generate Similarity Scores between q and r and current point of attention
                if self.use_attention:
                    question, _ = self.attention(question_t.reshape([-1, *question_t.shape]), rel.reshape(1, 1, -1))
                    question = question.squeeze()
                else:
                    question = question_t
                question = question.sum(dim=0)
                # Perceptron Module: Generate Semantic Score for action given q
                if self.use_perceptron:
                    question_with_history = self.perceptron4hq(torch.cat([history_t, question]))
                    score = (question_with_history * rel).sum()
                else:
                    rel = torch.norm(rel)
                    question = torch.norm(question)
                    score = (rel * question).sum()
                semantic_scores.append(score)
                actions.append(action)
        if len(actions) > 0:
            action_distribution = torch.softmax(torch.tensor(semantic_scores), dim=0)
            return actions, action_distribution
        return None, None
