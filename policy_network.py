import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.attention import Attention
from nets.perceptron import Perceptron
from expeiment_settings import ExperimentSettings

class PolicyNet(nn.Module):
    def __init__(self, use_attention, use_perceptron):
        super(PolicyNet, self).__init__()

        self.use_attention = use_attention
        self.use_perceptron = use_perceptron
        self.embedder = None

        self.attention_model = None
        self.perceptron = Perceptron(ExperimentSettings.dim, ExperimentSettings.dim, 2 * ExperimentSettings.dim)

    def forward(self, action_space, question_t: torch.Tensor, history_t: torch.Tensor):
        semantic_scores = []
        actions = []
        for action in action_space:
            for action in action_space:
                ## TODO: 提前load适应GPU
                rel = self.embedder.get_relation_embedding(action)
                if rel is not None:
                    # Attention Layer: Generate Similarity Scores between q and r and current point of attention
                    if self.use_attention:
                        question = self.attention(rel, question_t)
                    else:
                        question = question_t.sum(dim=0)

                    # Perceptron Module: Generate Semantic Score for action given q
                    if self.use_perceptron:
                        score = self.perceptron(rel, history_t, question)
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
