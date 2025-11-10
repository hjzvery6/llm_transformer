import argparse
import os
import json
import torch

def human_int(n):
    return f"{n:,}"

def main():
    parser = argparse.ArgumentParser(description="Inspect saved Transformer checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 顶层键
    print("=== Top-level keys ===")
    print(list(ckpt.keys()))
    print()

    # 训练信息
    epoch = ckpt.get("epoch", None)
    tr_loss = ckpt.get("train_loss", None)
    dv_loss = ckpt.get("dev_loss", None)
    ts_loss = ckpt.get("test_loss", None)
    tr_ppl = ckpt.get("train_ppl", None)
    dv_ppl = ckpt.get("dev_ppl", None)
    ts_ppl = ckpt.get("test_ppl", None)
    dv_bleu = ckpt.get("dev_bleu", None)
    ts_bleu = ckpt.get("test_bleu", None)

    print("=== Training Summary ===")
    print(f"epoch: {epoch}")
    print(f"train_loss: {tr_loss}")
    print(f"dev_loss:   {dv_loss}")
    print(f"test_loss:  {ts_loss}")
    print(f"train_ppl:  {tr_ppl}")
    print(f"dev_ppl:    {dv_ppl}")
    print(f"test_ppl:   {ts_ppl}")
    print(f"dev_bleu:   {dv_bleu}")
    print(f"test_bleu:  {ts_bleu}")

    print()

    # # 配置
    # config = ckpt.get("config", {})
    # print("=== Config ===")
    # print(json.dumps(config, ensure_ascii=False, indent=2))
    # print()

    # # 模型参数概览
    # model_state = ckpt.get("model_state", {})
    # total_params = sum(t.numel() for t in model_state.values())
    # print("=== Model State Dict ===")
    # print(f"num_tensors: {len(model_state)}")
    # print(f"total_params: {human_int(total_params)}")
    # # 打印前 10 个键及其形状
    # print("sample tensors (up to 10):")
    # for i, (k, v) in enumerate(model_state.items()):
    #     if i >= 10:
    #         break
    #     shape = tuple(v.shape)
    #     print(f" - {k}: {shape}")
    # print()

    # # 优化器信息
    # opt_state = ckpt.get("optimizer_state", {})
    # print("=== Optimizer State ===")
    # if "param_groups" in opt_state:
    #     param_groups = opt_state["param_groups"]
    #     print(f"param_groups: {len(param_groups)}")
    #     # 打印第一个组的关键参数
    #     g0 = param_groups[0]
    #     # 可能包含 keys: lr, betas, eps, weight_decay, initial_lr 等
    #     for key in ("lr", "betas", "eps", "weight_decay"):
    #         if key in g0:
    #             print(f" - {key}: {g0[key]}")
    # else:
    #     print("optimizer state does not contain 'param_groups'")
    # print()

    # print("Done.")

if __name__ == "__main__":
    main()