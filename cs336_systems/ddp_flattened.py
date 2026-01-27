import os
import time
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

# 引入你的基础模型
from cs336_basics.model import BasicsTransformerLM

# 引入三种 DDP 实现
# 1. Baseline Individual (Naive)
from cs336_systems.ddp_naive import NaiveDDP
# 2. Baseline Flattened 
from cs336_systems.ddp_flattened import MinimalDDPFlat
# 3. Overlap Individual (异步 overlap)
from cs336_systems.ddp_overlap_individual_parameters import DDP as DDPOverlapIndividual


# ----------------------------
# 1. 模型构建 (TODO 实现)
# ----------------------------
def build_xl_model(args, device: torch.device) -> nn.Module:
    """
    创建 XL 大小的模型。
    配置参考 benchmark.py 中的 MODEL_CONFIGS["xl"]。
    XL: d_model=1600, d_ff=6400, num_layers=48, num_heads=25
    """
    config = {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    }
    
    # 这里的 vocab_size 和 context_length 从 args 读取，
    # 确保和 benchmark 运行时参数一致
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        **config,
        rope_theta=10000.0, 
    )
    return model.to(device)


# ----------------------------
# 2. 数据生成 (TODO 实现)
# ----------------------------
def get_batch(device: torch.device, batch_size: int, seq_len: int, vocab_size: int):
    """
    生成 benchmark 用的随机数据。
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    return x, y


# ----------------------------
# 通用计时工具
# ----------------------------
@dataclass
class Timing:
    iter_ms: float
    comm_wait_ms: float


def _sync_and_time_ms(fn):
    """在 GPU 上准确计时：fn() 前后 torch.cuda.synchronize()"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def _allreduce_max_ms(value_ms: float, device: torch.device) -> float:
    """把 wall time 取跨 rank 的 max（迭代耗时由最慢 rank 决定）"""
    t = torch.tensor([value_ms], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


# ----------------------------
# 3. DDP Wrapper 工厂 (TODO 实现)
# ----------------------------
def make_ddp_wrapper(mode: str, model: nn.Module):
    """
    返回一个 wrapper 对象 w，满足：
      - w(*inputs, **kwargs) 做 forward
      - w.finish_gradient_synchronization() (可选) 用于在 step 前显式同步/等待梯度
    """

    if mode == "baseline-individual":
        # 1. 逐参数 AllReduce (无 Overlap)
        wrapper = NaiveDDP(model)
        # NaiveDDP 的同步方法叫 sync_gradients，适配 benchmark 调用的接口名
        wrapper.finish_gradient_synchronization = wrapper.sync_gradients
        return wrapper

    if mode == "baseline-flattened":
        # 2. Flatten 后单次 AllReduce (无 Overlap)
        wrapper = MinimalDDPFlat(model)
        # MinimalDDPFlat 的同步方法也叫 sync_gradients，进行适配
        wrapper.finish_gradient_synchronization = wrapper.sync_gradients
        return wrapper

    if mode == "overlap-individual":
        # 3. 逐参数 Async AllReduce (有 Overlap)
        # 这个类本身已经实现了 finish_gradient_synchronization
        return DDPOverlapIndividual(model)

    raise ValueError(f"Unknown mode: {mode}")


# ----------------------------
# 训练 step
# ----------------------------
def train_one_iter(wrapper, optimizer, loss_fn, batch):
    x, y = batch

    def _iter_body():
        optimizer.zero_grad(set_to_none=True)
        logits = wrapper(x)  
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        comm_wait_ms = 0.0
        # 如果 wrapper 需要显式同步（baseline 模式）或者等待 handle（overlap 模式）
        # 都在这里进行，并计入通信等待时间
        if hasattr(wrapper, "finish_gradient_synchronization"):
            comm_wait_ms = _sync_and_time_ms(wrapper.finish_gradient_synchronization)

        optimizer.step()
        return comm_wait_ms

    # 总 iter time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    comm_wait_ms = _iter_body()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    iter_ms = (t1 - t0) * 1000.0

    return Timing(iter_ms=iter_ms, comm_wait_ms=comm_wait_ms)


def run_benchmark(rank: int, world_size: int, args):
    # --- dist init ---
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", rank)

    # --- build model & wrapper ---
    # 传入 args 以获取 shape 信息
    model = build_xl_model(args, device=device)
    wrapper = make_ddp_wrapper(args.mode, model)

    # optimizer
    params = None
    if hasattr(wrapper, "module"):
        params = wrapper.module.parameters()
    elif hasattr(wrapper, "parameters"):
        params = wrapper.parameters()
    else:
        raise RuntimeError("Wrapper has no parameters() and no .module")

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    vocab_size = getattr(args, "vocab_size", 50304)

    # --- warmup ---
    if rank == 0:
        print(f"Warmup {args.warmup_iters} iters...")
    for _ in range(args.warmup_iters):
        batch = get_batch(device, args.batch_size, args.seq_len, vocab_size)
        _ = train_one_iter(wrapper, optimizer, loss_fn, batch)

    # --- timed iters ---
    if rank == 0:
        print(f"Benchmark {args.iters} iters...")
        
    iter_ms_list = []
    comm_ms_list = []
    
    for _ in range(args.iters):
        batch = get_batch(device, args.batch_size, args.seq_len, vocab_size)
        timing = train_one_iter(wrapper, optimizer, loss_fn, batch)

        # 统计：取所有 rank 中最慢的作为该次迭代的耗时（模拟同步步调）
        iter_ms = _allreduce_max_ms(timing.iter_ms, device)
        comm_ms = _allreduce_max_ms(timing.comm_wait_ms, device)
        iter_ms_list.append(iter_ms)
        comm_ms_list.append(comm_ms)

    # --- report on rank0 ---
    if rank == 0:
        import statistics as stats
        mean_iter = stats.mean(iter_ms_list)
        p50_iter = stats.median(iter_ms_list)
        p95_iter = stats.quantiles(iter_ms_list, n=20)[18]  # 95th
        mean_comm = stats.mean(comm_ms_list)

        print(f"\n=== Mode: {args.mode} | world_size={world_size} ===")
        print(f"iters={args.iters} warmup={args.warmup_iters}")
        print(f"batch_size={args.batch_size} seq_len={args.seq_len}")
        print(f"iter time (ms): mean={mean_iter:.3f}  p50={p50_iter:.3f}  p95={p95_iter:.3f}")
        print(f"comm-wait (ms): mean={mean_comm:.3f}")
        print("========================================\n")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline-individual", "baseline-flattened", "overlap-individual"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vocab-size", type=int, default=50304)

    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args()

    # 默认 2 卡测试
    world_size = 2
    if torch.cuda.device_count() < world_size:
        print(f"Warning: Not enough GPUs ({torch.cuda.device_count()} found), forcing world_size=1 for debugging.")
        world_size = 1

    mp.spawn(run_benchmark, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()