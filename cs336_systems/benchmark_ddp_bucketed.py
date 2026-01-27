# -*- coding: utf-8 -*-
"""
Problem (ddp_bucketed_benchmarking) - (a)
Benchmark bucketed DDP with bucket sizes in {1, 10, 100, 1000} MB
on 1 node, 2 GPUs, XL model config.

Run:
  torchrun --nproc_per_node=2 cs336_systems/ddp_bucketed_benchmarking.py \
    --model-size xl --context-len 128 --batch-size 4 \
    --bucket-sizes-mb 1 10 100 1000 --steps 30 --warmup 10 \
    --out-csv ddp_bucketed_benchmark.csv

Notes:
- This script assumes you already implemented bucketed DDP in Problem (ddp_overlap_bucketed).
- It tries several import paths / class names for compatibility.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


# -------------------------
# 1) Model config (XL)
# -------------------------
@dataclass(frozen=True)
class ModelCfg:
    vocab_size: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS = {
    # 按 CS336 常见配置写；如果你仓库里 xl 定义不同，改这里即可。
    "xl": ModelCfg(vocab_size=50257, d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
}


def build_model(model_size: str, device: torch.device) -> torch.nn.Module:
    """
    兼容两种常见仓库结构：
    - cs336_basics.model.BasicsTransformerLM
    - cs336_basics.models 或其它（你可按需改 import）
    """
    cfg = MODEL_CONFIGS[model_size]

    try:
        from cs336_basics.model import BasicsTransformerLM  # 常见命名
    except Exception as e:
        raise RuntimeError(
            "找不到 cs336_basics.model.BasicsTransformerLM。"
            "请按你仓库的真实路径修改 build_model() 里的 import。"
        ) from e

    # 下面参数名也可能与你仓库不一致：若报错，按你 BasicsTransformerLM 的 __init__ 改即可
    model = BasicsTransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=args.context_len,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
    ).to(device)

    return model


# -------------------------
# 2) Import bucketed DDP
# -------------------------
def get_bucketed_ddp(module: torch.nn.Module, bucket_size_mb: float):
    """
    尝试导入你在 Problem (ddp_overlap_bucketed) 中写的实现。
    这里做“多候选路径/类名”的兼容，避免你仓库命名不同导致脚本不能跑。
    """
    candidates = [
        # 常见：cs336_systems/ddp_overlap_bucketed.py 里 class DDP
        ("cs336_systems.ddp_overlap_bucketed", "DDP"),
        # 也有人会叫 BucketedDDP
        ("cs336_systems.ddp_overlap_bucketed", "BucketedDDP"),
        # 或文件名叫 ddp_bucketed
        ("cs336_systems.ddp_bucketed", "DDP"),
        ("cs336_systems.ddp_bucketed", "BucketedDDP"),
    ]

    last_err: Optional[Exception] = None
    for mod_name, cls_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            return cls(module, bucket_size_mb=bucket_size_mb)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "无法导入 bucketed DDP 实现。你需要把 get_bucketed_ddp() 里的 candidates\n"
        "改成你仓库中实际的模块路径和类名。例如：('cs336_systems.ddp_overlap_bucketed', 'DDP')。\n"
        f"最后一次错误: {last_err}"
    )


# -------------------------
# 3) DDP init / cleanup
# -------------------------
def init_dist_from_torchrun() -> Tuple[int, int, torch.device]:
    """
    torchrun 会设置：
      LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed 不可用")
    if dist.is_initialized():
        raise RuntimeError("process group 已初始化；不要重复 init")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, device


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# 4) One benchmark run
# -------------------------
@torch.no_grad()
def _sync_barrier():
    # 让 timing 更稳定：rank 间对齐 + CUDA 同步
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_steps(
    ddp,
    vocab_size: int,
    batch_size: int,
    context_len: int,
    steps: int,
    warmup: int,
    lr: float,
    device: torch.device,
) -> List[float]:
    """
    返回每一步迭代的 wall-clock time（秒），只记录 warmup 之后的 steps。
    """
    # 用 wrapped module 的参数（你的 DDP 里一般是 ddp.module）
    params = ddp.module.parameters() if hasattr(ddp, "module") else ddp.parameters()
    opt = torch.optim.AdamW(params, lr=lr)

    times: List[float] = []

    for it in range(warmup + steps):
        _sync_barrier()
        t0 = time.perf_counter()

        # 造一批随机 token，做 next-token 预测
        x = torch.randint(0, vocab_size, (batch_size, context_len), device=device)
        # targets: shift-left
        y = x[:, 1:].contiguous()

        opt.zero_grad(set_to_none=True)

        logits = ddp.forward(x)  # [B, T, V]（常见）
        # 对齐 next-token：预测 x[t+1] 用 logits[t]
        logits = logits[:, :-1, :].contiguous()

        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()

        # 关键：等待 bucketed async all-reduce 都排队/完成（按你实现）
        ddp.finish_gradient_synchronization()

        opt.step()

        _sync_barrier()
        t1 = time.perf_counter()

        if it >= warmup:
            times.append(t1 - t0)

    return times


def reduce_mean(x: float, device: torch.device) -> float:
    t = torch.tensor([x], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


# -------------------------
# 5) Main
# -------------------------
def main(args):
    rank, world_size, device = init_dist_from_torchrun()
    assert world_size == args.world_size, (
        f"world_size 不一致：torchrun={world_size}, --world-size={args.world_size}"
    )

    if rank == 0:
        print(f"[init] world_size={world_size}, device={device}, model_size={args.model_size}")
        print(f"[init] bucket_sizes_mb={args.bucket_sizes_mb}")

    # 构建模型
    cfg = MODEL_CONFIGS[args.model_size]
    model = build_model(args.model_size, device=device)

    # 输出 CSV：只让 rank0 写
    if rank == 0 and args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "bucket_size_mb",
                    "mean_iter_ms(avg_over_ranks)",
                    "p50_iter_ms(avg_over_ranks)",
                    "p90_iter_ms(avg_over_ranks)",
                ]
            )

    for bucket_mb in args.bucket_sizes_mb:
        # 重新 wrap（避免 bucket 配置残留）
        ddp = get_bucketed_ddp(model, bucket_size_mb=float(bucket_mb))

        # 跑一次
        times = run_steps(
            ddp=ddp,
            vocab_size=cfg.vocab_size,
            batch_size=args.batch_size,
            context_len=args.context_len,
            steps=args.steps,
            warmup=args.warmup,
            lr=args.lr,
            device=device,
        )

        # 统计（每个 rank 先算自己的 p50/p90/mean，然后对 rank 取平均；更稳定）
        times_sorted = sorted(times)
        mean_s = sum(times) / len(times)
        p50_s = times_sorted[int(0.50 * (len(times_sorted) - 1))]
        p90_s = times_sorted[int(0.90 * (len(times_sorted) - 1))]

        mean_ms = reduce_mean(mean_s * 1e3, device=device)
        p50_ms = reduce_mean(p50_s * 1e3, device=device)
        p90_ms = reduce_mean(p90_s * 1e3, device=device)

        if rank == 0:
            print(
                f"[bucket={bucket_mb:>5}MB] mean={mean_ms:.3f} ms | "
                f"p50={p50_ms:.3f} ms | p90={p90_ms:.3f} ms"
            )
            if args.out_csv:
                with open(args.out_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([bucket_mb, f"{mean_ms:.6f}", f"{p50_ms:.6f}", f"{p90_ms:.6f}"])

    cleanup_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2, help="题目要求 2 GPUs（torchrun 也要一致）")
    parser.add_argument("--model-size", type=str, default="xl", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--bucket-sizes-mb",
        type=float,
        nargs="+",
        default=[1, 10, 100, 1000],
        help="题目要求扫 1/10/100/1000 MB",
    )
    parser.add_argument("--out-csv", type=str, default="ddp_bucketed_benchmark.csv")
    args = parser.parse_args()

    main(args)