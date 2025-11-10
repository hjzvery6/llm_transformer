import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForwardNetwork(nn.Module):
    """
    位置前馈网络：每个位置独立地通过两层全连接网络。
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x