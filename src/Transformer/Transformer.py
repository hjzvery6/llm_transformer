import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .PositionEncoding import PositionalEncoding
from .EncoderLayer import EncoderLayer
from .DecoderLayer import DecoderLayer

class Encoder(nn.Module):
    """
    编码器：N个EncoderLayer堆叠
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)  # 缩放嵌入
        x = self.pos_encoding(x)
        x = self.dropout(x)
        # 遍历多个EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    """
    解码器：N个DecoderLayer堆叠而成
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)  # 缩放嵌入
        x = self.pos_encoding(x)
        x = self.dropout(x)
        # 遍历多个DecoderLayer
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    """
    Transformer模型：编码器、解码器、解码器输出投影
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 d_ff=2048, num_layers=6, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src: 源序列，形状 (batch_size, src_len)
        tgt: 目标序列，形状 (batch_size, tgt_len)
        src_mask: 源序列掩码（用于忽略 padding）
        tgt_mask: 目标序列掩码（用于防止看到未来词）
        """
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.output_proj(dec_output)  # (batch_size, tgt_len, tgt_vocab_size)
        return output

