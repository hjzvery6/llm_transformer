import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .MultiHeadAttention import MultiHeadAttention
from .FFN import FeedForwardNetwork

class DecoderLayer(nn.Module):
    """
    解码器的一层：包含掩码多头自注意力（防止看到未来信息）、编码器-解码器注意力（关注编码器输出）、前馈网络
    每部分都有残差连接和层归一化。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 掩码多头自注意力
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        # 残差连接 + 层归一化
        x = self.norm1(x + self.dropout1(self_attn_output))

        # 交叉注意力
        enc_dec_attn_output, _ = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        # 残差连接 + 层归一化
        x = self.norm2(x + self.dropout2(enc_dec_attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        # 残差连接 + 层归一化
        x = self.norm3(x + self.dropout3(ff_output))
        return x