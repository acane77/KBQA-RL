import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=out_features)

    def reset_parameter(self):
        nn.init.xavier_normal_(self.fc1)
        nn.init.xavier_normal_(self.fc2)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        return out
