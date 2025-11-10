import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .MultiHeadAttention import MultiHeadAttention
from .FFN import FeedForwardNetwork

class EncoderLayer(nn.Module):
    """
    编码器的一层：包含多头自注意力 + 前馈网络
    每部分都有残差连接和层归一化
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 多头自注意力
        attn_output, _ = self.self_attn(x, x, x, mask)
        # 残差连接 + 层归一化
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        # 残差连接 + 层归一化
        x = self.norm2(x + self.dropout2(ff_output))
        return x