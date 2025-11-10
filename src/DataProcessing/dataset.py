import os
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from typing import List, Tuple

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3

def create_padding_mask(seq: torch.Tensor, pad_token: int = PAD_ID) -> torch.Tensor:
    """
    功能：为序列创建 padding 掩码，屏蔽掉填充位置（pad_token）
    输入：
    seq: 形状 (batch, seq_len) 的整型张量，值为子词 ID
    pad_token: 用作 padding 的 ID（默认 0）

    输出：
    掩码张量，形状 (batch, 1, 1, seq_len)，True 表示“有效（非 pad）”，False 表示“无效（pad）”

    说明：
    这种形状便于在注意力分数矩阵上做广播，屏蔽掉被 pad 的位置
    """
    # (batch, seq_len) -> (batch, 1, 1, seq_len)
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size: int) -> torch.Tensor:
    """
    功能：为解码器自注意力创建掩码，禁止看到未来词
    输入：
    size: 目标序列长度（T）

    输出：
    掩码张量，形状 (1, 1, T, T)，下三角（含对角线）为 True，其余为 False

    说明：
    与解码端的自注意力结合使用，以实现自回归训练（每个位置仅可看见自己及之前的词）
    """
    # (size, size) 下三角 True
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_token: int = PAD_ID) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    功能：根据源序列与目标序列，生成训练所需的掩码
    输入：
    src: 源序列张量，形状 (batch, S)
    tgt: 目标“解码输入”张量，形状 (batch, T)，通常为 [BOS] + tgt_ids
    pad_token: padding 的 ID（默认 0）

    输出：
    src_mask: 源掩码，形状 (batch, 1, 1, S)，True=有效，False=pad
    tgt_mask: 目标掩码，形状 (batch, 1, T, T)，True=可见，False=不可见（pad 或未来）

    说明：
    tgt_mask = tgt_pad_mask & tgt_look_ahead
    同时屏蔽 pad 和未来位置，使解码器自注意力只关注已有的有效 token
    """
    src_mask = create_padding_mask(src, pad_token)
    tgt_pad_mask = create_padding_mask(tgt, pad_token)
    tgt_look_ahead = create_look_ahead_mask(tgt.size(1))
    tgt_mask = tgt_pad_mask & tgt_look_ahead
    return src_mask, tgt_mask

class IWSLTDataset(Dataset):
    """
    IWSLT英德平行语料的数据集封装：
    读取处理后的文本行（src_txt/tgt_txt）
    使用已训练好的 SentencePiece 模型把文本编码为子词 ID
    可选地按 max_len 过滤过长样本，减少 GPU 显存占用与序列截断
    构造解码器训练所需的 (tgt_inp, tgt_out) 对（右移一位）
    """
    def __init__(self, src_txt: str, tgt_txt: str, spm_src_model: str, spm_tgt_model: str, max_len: int = 128):
        # 读取源/目标文本，每行一个句子；要求两侧样本数一致
        self.src_lines = self._read_lines(src_txt)
        self.tgt_lines = self._read_lines(tgt_txt)
        assert len(self.src_lines) == len(self.tgt_lines), "英德样本数不一致"

        # 加载 SentencePiece 分词器（src/tgt分别对应 英语/德语）
        self.sp_src = spm.SentencePieceProcessor(model_file=spm_src_model)
        self.sp_tgt = spm.SentencePieceProcessor(model_file=spm_tgt_model)
        self.max_len = max_len

        # 预编码并长度过滤：
        # 先把每个句子编码为子词 ID
        # 丢弃空样本与过长样本（> max_len）
        # 保留 (src_ids, tgt_ids) 到 self.pairs
        self.pairs = []
        for en, de in zip(self.src_lines, self.tgt_lines):
            src_ids = self.sp_src.encode(en, out_type=int)
            tgt_ids = self.sp_tgt.encode(de, out_type=int)
            if len(src_ids) == 0 or len(tgt_ids) == 0:
                continue
            if len(src_ids) > max_len or len(tgt_ids) > max_len:
                continue
            self.pairs.append((src_ids, tgt_ids))

    def _read_lines(self, fp: str) -> List[str]:
        # 简单读取，去掉空行；返回清洗后的句子列表
        with open(fp, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        # 返回样本数（经过编码与过滤后的）
        return len(self.pairs)

    def __getitem__(self, idx: int):
        # 取第 idx 个样本，构造解码端输入/输出：
        # tgt_inp: [BOS] + tgt_ids      （网络输入）
        # tgt_out: tgt_ids + [EOS]      （损失对齐目标）
        src_ids, tgt_ids = self.pairs[idx]
        tgt_inp = [BOS_ID] + tgt_ids
        tgt_out = tgt_ids + [EOS_ID]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_inp, dtype=torch.long), torch.tensor(tgt_out, dtype=torch.long)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    DataLoader 的聚合函数：
    接收一个 batch 的样本列表 (src_ids, tgt_inp, tgt_out)
    进行动态 padding：对齐到本 batch 的最大长度，pad 到 `PAD_ID`
    构造与 Transformer 兼容的掩码：`src_mask` 与 `tgt_mask`

    返回：
    src_batch: (B, S_max)
    tgt_inp_batch: (B, T_max)
    tgt_out_batch: (B, T_max)
    src_mask: (B, 1, 1, S_max)
    gt_mask: (B, 1, T_max, T_max)
    """
    # 动态 padding
    src_seqs, tgt_inps, tgt_outs = zip(*batch)
    src_max = max(x.size(0) for x in src_seqs)
    tgt_max = max(x.size(0) for x in tgt_inps)

    def pad_to(seqs: List[torch.Tensor], max_len: int):
        # 把每个序列 pad 到 max_len，pad 值为 PAD_ID；最后堆叠成 (batch, max_len)
        out = []
        for s in seqs:
            pad_len = max_len - s.size(0)
            if pad_len > 0:
                s = torch.cat([s, torch.full((pad_len,), PAD_ID, dtype=torch.long)])
            out.append(s)
        return torch.stack(out, dim=0)  # (batch, max_len)

    src_batch = pad_to(src_seqs, src_max)
    tgt_inp_batch = pad_to(tgt_inps, tgt_max)
    tgt_out_batch = pad_to(tgt_outs, tgt_max)

    # 掩码：源端只考虑非 pad；目标端既考虑非 pad 又考虑前瞻屏蔽
    src_mask, tgt_mask = create_masks(src_batch, tgt_inp_batch, PAD_ID)
    return src_batch, tgt_inp_batch, tgt_out_batch, src_mask, tgt_mask