import os
import time
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


# ----------------------------
# TODO: 你需要把这两个函数对接到你仓库已有代码
# ----------------------------
def build_xl_model(device: torch.device) -> nn.Module:
    """
    TODO: 改成你仓库里创建 XL 模型的方式（§1.1.2 的 XL size）。
    例如（伪代码）:
      from cs336_systems.models import make_model
      model = make_model(size="xl").to(device)
      return model
    """
    raise NotImplementedError("Please wire build_xl_model() to your repo's XL model builder.")


def get_batch(device: torch.device, batch_size: int, seq_len: int, vocab_size: int):
    """
    TODO: 改成你仓库里 benchmark 用的数据生成方式。
    这里给一个通用 LM 假数据：input_ids + targets
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
# 三种 DDP wrapper 的“适配接口”
# 你可以把这些 import 换成你仓库里已经实现的类/函数
# ----------------------------
def make_ddp_wrapper(mode: str, model: nn.Module):
    """
    返回一个 wrapper 对象 w，满足：
      - w(*inputs, **kwargs) 做 forward
      - w.finish_gradient_synchronization() (可选) 用于 overlap 模式在 step 前 wait handles
      - w.module (可选) 指向原始模型（给 optimizer 用也行）
    """
    # 你仓库里大概率有 adapters / ddp_xxx 的实现；这里按“常见作业结构”写：
    #   from cs336_systems import adapters
    #   w = adapters.get_ddp_individual_parameters(model, overlap=...)
    #
    # 你只要把下面三段替换成你真实的构造函数即可。

    if mode == "baseline-individual":
        # TODO: 替换成你“每参数一次 all_reduce（不 overlap）”的 DDP 实现
        # e.g. return adapters.get_ddp_individual_parameters(model, overlap=False)
        raise NotImplementedError

    if mode == "baseline-flattened":
        # TODO: 替换成你“flatten grads 一次 all_reduce（不 overlap）”的实现
        # e.g. return adapters.get_ddp_flattened(model)
        raise NotImplementedError

    if mode == "overlap-individual":
        # TODO: 替换成你“hook + async all_reduce（overlap）”的实现
        # e.g. return adapters.get_ddp_individual_parameters(model, overlap=True)
        raise NotImplementedError

    raise ValueError(f"Unknown mode: {mode}")


# ----------------------------
# 训练 step（统计总耗时 + 通信等待耗时）
# ----------------------------
def train_one_iter(wrapper, optimizer, loss_fn, batch):
    x, y = batch

    def _iter_body():
        optimizer.zero_grad(set_to_none=True)
        logits = wrapper(x)  # 假设 model forward 只吃 x，输出 logits
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        comm_wait_ms = 0.0
        # overlap 模式：通常你需要在 step 前等所有 async allreduce “queued/ready”
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
    model = build_xl_model(device=device)
    wrapper = make_ddp_wrapper(args.mode, model)

    # optimizer：一般用 wrapper.module.parameters() 或 wrapper.parameters()
    params = None
    if hasattr(wrapper, "module"):
        params = wrapper.module.parameters()
    elif hasattr(wrapper, "parameters"):
        params = wrapper.parameters()
    else:
        raise RuntimeError("Wrapper has no parameters() and no .module")

    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # loss：假设是 LM logits
    loss_fn = nn.CrossEntropyLoss()

    # 你也可以从 model/config 里拿 vocab_size，这里给个默认值
    vocab_size = getattr(args, "vocab_size", 50304)

    # --- warmup ---
    for _ in range(args.warmup_iters):
        batch = get_batch(device, args.batch_size, args.seq_len, vocab_size)
        _ = train_one_iter(wrapper, optimizer, loss_fn, batch)

    # --- timed iters ---
    iter_ms_list = []
    comm_ms_list = []
    for _ in range(args.iters):
        batch = get_batch(device, args.batch_size, args.seq_len, vocab_size)
        timing = train_one_iter(wrapper, optimizer, loss_fn, batch)

        # 取跨 rank max
        iter_ms = _allreduce_max_ms(timing.iter_ms, device)
        comm_ms = _allreduce_max_ms(timing.comm_wait_ms, device)
        iter_ms_list.append(iter_ms)
        comm_ms_list.append(comm_ms)

    # --- report on rank0 ---
    if rank == 0:
        import statistics as stats
        mean_iter = stats.mean(iter_ms_list)
        p50_iter = stats.median(iter_ms_list)
        p95_iter = stats.quantiles(iter_ms_list, n=20)[18]  # 95th ≈ 19/20
        mean_comm = stats.mean(comm_ms_list)

        print(f"\n=== Mode: {args.mode} | world_size={world_size} ===")
        print(f"iters={args.iters} warmup={args.warmup_iters}")
        print(f"iter time (ms): mean={mean_iter:.3f}  p50={p50_iter:.3f}  p95={p95_iter:.3f}")
        print(f"comm-wait (ms): mean={mean_comm:.3f}  (only meaningful if wrapper.finish_gradient_synchronization exists)")
        print("========================================\n")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline-individual", "baseline-flattened", "overlap-individual"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vocab-size", type=int, default=50304)

    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args()

    world_size = 2
    mp.spawn(run_benchmark, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
