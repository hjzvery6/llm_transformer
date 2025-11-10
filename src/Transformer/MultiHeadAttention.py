import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制：
    将 Q、K、V 投影到多个子空间，分别计算注意力，再拼接结果。
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # 定义线性变换层（用于生成 Q, K, V 和输出投影）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力：
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (..., seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 将 mask 为 0 的位置设为极小值
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换并分头：(batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头：(batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 最终线性投影
        output = self.W_o(output)
        return output, attn_weights