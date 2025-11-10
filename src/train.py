import os
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from Transformer.Transformer import Transformer
from DataProcessing.dataset import IWSLTDataset, collate_fn, PAD_ID
import sentencepiece as spm
import sacrebleu
from DataProcessing.dataset import BOS_ID, EOS_ID, create_padding_mask

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.opt = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5 *
              min(self.step_num ** -0.5, self.step_num * self.warmup ** -1.5))
        for group in self.opt.param_groups:
            group['lr'] = lr
        return lr

def set_seed(seed: int = 42):
    # 设定所有随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_paths():
    # 统一管理各类路径，方便在代码其余部分引用
    base_dir = os.path.dirname(__file__)  # 当前文件所在目录
    proc_dir = os.path.join(base_dir, "DataProcessing", "processed_data")
    paths = {
        # 原始文本
        "train_en": os.path.join(proc_dir, "train.en"),
        "train_de": os.path.join(proc_dir, "train.de"),
        # "train_en": os.path.join(proc_dir, "train_small.en"),
        # "train_de": os.path.join(proc_dir, "train_small.de"),
        "dev_en": os.path.join(proc_dir, "dev.en"),
        "dev_de": os.path.join(proc_dir, "dev.de"),
        # "test_en": os.path.join(proc_dir, "tst2014.en"),
        # "test_de": os.path.join(proc_dir, "tst2014.de"),
        "test_en": os.path.join(proc_dir, "test.en"),
        "test_de": os.path.join(proc_dir, "test.de"),
        # SentencePiece 模型（英文/德文）
        "spm_en": os.path.join(proc_dir, "spm_en.model"),
        "spm_de": os.path.join(proc_dir, "spm_de.model"),
        # Checkpoint 保存目录、训练曲线图片路径
        "ckpt_dir": os.path.join(base_dir, "Checkpoints"),
        "fig_path": os.path.join(base_dir, "../results/training_curve.png"),
    }
    # 保证 checkpoint 目录存在
    os.makedirs(paths["ckpt_dir"], exist_ok=True)
    return paths

def build_vocab_sizes(spm_en_path: str, spm_de_path: str):
    # 加载 SentencePiece 模型并读取词表大小
    # src_vocab_size：英文词表大小
    # tgt_vocab_size：德文词表大小
    sp_en = spm.SentencePieceProcessor(model_file=spm_en_path)
    sp_de = spm.SentencePieceProcessor(model_file=spm_de_path)
    src_vocab_size = sp_en.get_piece_size()
    tgt_vocab_size = sp_de.get_piece_size()
    return src_vocab_size, tgt_vocab_size

def build_loaders(paths, max_len=128, batch_size=64):
    # 构建数据集与 DataLoader
    # IWSLTDataset：
    # 负责从文本加载样本，使用 SentencePiece 对 src/tgt 分词与数字化
    # 应该会在 collate_fn 中返回：
    # src_batch        : (B, S) 源序列 token id
    # tgt_inp_batch    : (B, T) 解码器输入（通常为 <bos> + tgt[:-1]）
    # tgt_out_batch    : (B, T) 预测目标（通常为 tgt）
    # src_mask         : 源序列 mask（屏蔽 padding）
    # tgt_mask         : 目标序列 mask（包含 padding mask 与未来时刻的下三角掩码）
    train_ds = IWSLTDataset(
        src_txt=paths["train_en"], tgt_txt=paths["train_de"],
        spm_src_model=paths["spm_en"], spm_tgt_model=paths["spm_de"],
        max_len=max_len
    )
    dev_ds = IWSLTDataset(
        src_txt=paths["dev_en"], tgt_txt=paths["dev_de"],
        spm_src_model=paths["spm_en"], spm_tgt_model=paths["spm_de"],
        max_len=max_len
    )
    test_ds = IWSLTDataset(
        src_txt=paths["test_en"], tgt_txt=paths["test_de"],
        spm_src_model=paths["spm_en"], spm_tgt_model=paths["spm_de"],
        max_len=max_len
    )
    # 数据集的DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    return train_loader, dev_loader, test_loader

# 解码与BLEU工具函数
def subsequent_mask(sz: int, device):
    # 下三角因果遮罩 (1, 1, L, L)
    mask = torch.tril(torch.ones((sz, sz), device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(1)

def create_tgt_mask(tgt_seq: torch.Tensor):
    # 结合 padding mask 与因果遮罩
    # tgt_seq: (B, L)
    device = tgt_seq.device
    pad_mask = create_padding_mask(tgt_seq, PAD_ID)  # (B, 1, 1, L)
    causal = subsequent_mask(tgt_seq.size(1), device)  # (1, 1, L, L)
    return pad_mask & causal  # (B, 1, L, L) 广播得到

def strip_special(ids: torch.Tensor):
    # 去掉 BOS/PAD，并截断到 EOS
    out = []
    for t in ids.tolist():
        if t == EOS_ID:
            break
        if t != PAD_ID and t != BOS_ID:
            out.append(t)
    return out

@torch.no_grad()
def greedy_decode(model, src_batch, src_mask, max_len):
    # 贪心生成目标序列 (B, <=max_len)
    device = src_batch.device
    B = src_batch.size(0)
    ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = create_tgt_mask(ys)
        logits = model(src_batch, ys, src_mask, tgt_mask)  # (B, L, V)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        ys = torch.cat([ys, next_token], dim=1)
        finished |= (next_token.squeeze(1) == EOS_ID)
        if finished.all():
            break
    return ys  # (B, L_gen)

@torch.no_grad()
def compute_bleu(model, loader, device, spm_de, max_len):
    # 生成 + 反解码 + corpus BLEU
    model.eval()
    hyps, refs = [], []
    for src_batch, tgt_inp_batch, tgt_out_batch, src_mask, tgt_mask in loader:
        src_batch = src_batch.to(device)
        tgt_out_batch = tgt_out_batch.to(device)
        src_mask = src_mask.to(device)

        gen_ids = greedy_decode(model, src_batch, src_mask, max_len)  # (B, L)
        for i in range(src_batch.size(0)):
            hyp_ids = strip_special(gen_ids[i])
            ref_ids = strip_special(tgt_out_batch[i])
            hyp_text = spm_de.decode(hyp_ids) if len(hyp_ids) > 0 else ""
            ref_text = spm_de.decode(ref_ids) if len(ref_ids) > 0 else ""
            hyps.append(hyp_text)
            refs.append(ref_text)
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score if len(hyps) > 0 else 0.0
    return bleu

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, tgt_vocab_size, grad_clip):
    # 单个 epoch 的训练过程：
    # 对每个 batch 前向、计算损失、反向、梯度裁剪、更新参数
    # 统计平均损失（按 batch 平均）与处理 token 数（不含 PAD）
    model.train()
    total_loss_sum = 0.0      # 累积的 batch 平均损失
    total_tokens = 0      # 统计非 PAD 的目标 token 数
    start = time.time()

    for src_batch, tgt_inp_batch, tgt_out_batch, src_mask, tgt_mask in loader:
        # 将所有张量移动到目标设备（CPU/GPU）
        src_batch = src_batch.to(device)
        tgt_inp_batch = tgt_inp_batch.to(device)
        tgt_out_batch = tgt_out_batch.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        optimizer.zero_grad()

        # 模型前向：
        # logits 形状 (B, T, V)，V 为目标词表大小
        logits = model(src_batch, tgt_inp_batch, src_mask, tgt_mask)

        # 交叉熵损失：忽略 PAD
        # - 将 (B, T, V) 展成 (B*T, V)，目标展成 (B*T)
        # - criterion(ignore_index=PAD_ID) 生效于 PAD 位置
        loss = criterion(
            logits.reshape(-1, tgt_vocab_size),
            tgt_out_batch.reshape(-1)
        )

        # 反向与优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 防止梯度爆炸
        optimizer.step()
        scheduler.step()

        valid_tokens = (tgt_out_batch != PAD_ID).sum().item()
        total_tokens += valid_tokens
        total_loss_sum += loss.item() * valid_tokens

    elapsed = time.time() - start
    avg_loss = total_loss_sum / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 10 else float("inf")
    return avg_loss, ppl, elapsed, total_tokens

@torch.no_grad()
def evaluate(model, loader, criterion, device, tgt_vocab_size):
    # 在验证集和测试集上评估：
    # 不计算梯度（@torch.no_grad()）
    # 与训练类似的前向与损失计算，但不做反向与更新
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0

    for src_batch, tgt_inp_batch, tgt_out_batch, src_mask, tgt_mask in loader:
        src_batch = src_batch.to(device)
        tgt_inp_batch = tgt_inp_batch.to(device)
        tgt_out_batch = tgt_out_batch.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        logits = model(src_batch, tgt_inp_batch, src_mask, tgt_mask)
        loss = criterion(
            logits.reshape(-1, tgt_vocab_size),
            tgt_out_batch.reshape(-1)
        )

        valid_tokens = (tgt_out_batch != PAD_ID).sum().item()
        total_tokens += valid_tokens
        total_loss_sum += loss.item() * valid_tokens

    avg_loss = total_loss_sum / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 10 else float("inf")
    return avg_loss, ppl

def plot_curves(train_losses, dev_losses, fig_path):
    # 绘制并保存训练/验证损失随 epoch 变化的曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="tab:blue")
    plt.plot(dev_losses, label="Test Loss", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (CrossEntropy)")
    plt.title("Transformer EN→DE Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"训练曲线已保存到: {fig_path}")

def main():
    # 固定随机种子与设备
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 训练配置（支持命令行参数覆盖）
    parser = argparse.ArgumentParser(description="Train Transformer EN→DE")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--bleu_every", type=int, default=1)
    args = parser.parse_args()

    # 训练配置（可按需调整）
    epochs = args.epochs
    batch_size = args.batch_size
    max_len = args.max_len
    d_model = args.d_model
    num_heads = args.num_heads
    d_ff = args.d_ff
    num_layers = args.num_layers
    dropout = args.dropout
    lr = args.lr
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    label_smoothing = args.label_smoothing
    grad_clip = args.grad_clip
    bleu_every = args.bleu_every

    # 路径与词表大小
    paths = get_paths()
    src_vocab_size, tgt_vocab_size = build_vocab_sizes(paths["spm_en"], paths["spm_de"])
    print(f"Vocab sizes -> src: {src_vocab_size}, tgt: {tgt_vocab_size}")
    spm_de = spm.SentencePieceProcessor(model_file=paths["spm_de"])

    # 构建Dataloader
    train_loader, dev_loader, test_loader = build_loaders(paths, max_len=max_len, batch_size=batch_size)
    print(f"Train batches: {len(train_loader)}, Dev batches: {len(dev_loader)}, Test batches: {len(test_loader)}")

    # 构建模型、优化器与损失函数
    model = Transformer(
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, dropout=dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing)  # 忽略 PAD 的位置

    train_losses, dev_losses, test_losses = [], [], []

    # 训练循环：每个 epoch 训练 + 在验证集和测试集评估 + 保存 checkpoint
    for epoch in range(1, epochs + 1):
        # 训练一个 epoch
        tr_loss, tr_ppl, tr_time, tr_tokens = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, tgt_vocab_size, grad_clip
        )
        # 在验证集评估
        dv_loss, dv_ppl = evaluate(model, dev_loader, criterion, device, tgt_vocab_size)
        # 在测试集评估
        ts_loss, ts_ppl = evaluate(model, test_loader, criterion, device, tgt_vocab_size)

        # 计算 BLEU（按频率）
        if epoch % bleu_every == 0:
            dv_bleu = compute_bleu(model, dev_loader, device, spm_de, max_len)
            ts_bleu = compute_bleu(model, test_loader, device, spm_de, max_len)
        else:
            dv_bleu = None
            ts_bleu = None

        train_losses.append(tr_loss)
        dev_losses.append(dv_loss)
        test_losses.append(ts_loss)

        # 打印当前 epoch 的关键指标
        print(f"[Epoch {epoch:1d}] "
              f"train_loss={tr_loss:.4f} train_ppl={tr_ppl:.2f} "
              f"dev_loss={dv_loss:.4f} dev_ppl={dv_ppl:.2f} dev_bleu={(dv_bleu if dv_bleu is not None else 0):.2f} "
              f"test_loss={ts_loss:.4f} test_ppl={ts_ppl:.2f} test_bleu={(ts_bleu if ts_bleu is not None else 0):.2f} "
            #   f"dev_bleu={(dv_bleu if dv_bleu is not None else 0):.2f} "
            #   f"test_bleu={(ts_bleu if ts_bleu is not None else 0):.2f} "
              f"time={tr_time:.1f}s tokens={tr_tokens}")

        # 保存 checkpoint（包含模型、优化器状态和配置）
        ckpt_path = os.path.join(paths["ckpt_dir"], f"transformer_en_de_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "src_vocab_size": src_vocab_size, "tgt_vocab_size": tgt_vocab_size,
                "d_model": d_model, "num_heads": num_heads, "d_ff": d_ff, "num_layers": num_layers,
                "dropout": dropout, "max_len": max_len
            },
            "train_loss": tr_loss, "dev_loss": dv_loss, "test_loss": ts_loss,
            "train_ppl": tr_ppl, "dev_ppl": dv_ppl, "test_ppl": ts_ppl,
            "dev_bleu": dv_bleu, "test_bleu": ts_bleu
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    # 绘制训练曲线
    plot_curves(train_losses, test_losses, paths["fig_path"])

if __name__ == "__main__":
    main()