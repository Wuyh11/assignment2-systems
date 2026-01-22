import os
import time
import socket
import argparse
import statistics as stats

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _pick_free_port() -> int:
    """Pick a free port on localhost (best-effort)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def setup_process_group(rank: int, world_size: int, backend: str, master_addr: str, master_port: int):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    # For single node spawn, rank is global rank.
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


def benchmark_worker(rank: int, args, master_port: int):
    # 1) init
    setup_process_group(
        rank=rank,
        world_size=args.world_size,
        backend=args.backend,
        master_addr=args.master_addr,
        master_port=master_port,
    )

    # 2) device
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available but device=cuda was requested."
        torch.cuda.set_device(rank)  # single-node multi-GPU: rank -> local GPU id
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    # 3) tensor to all-reduce
    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.numel, device=device, dtype=dtype)

    # optional: make comm a bit more realistic (avoid all ranks having identical input)
    x += (rank + 1) * 0.001

    # 4) warmup
    for _ in range(args.warmup):
        if args.device == "cuda":
            torch.cuda.synchronize()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)
        if args.device == "cuda":
            torch.cuda.synchronize()

    # 5) timed iterations
    times_ms = []
    for _ in range(args.iters):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)

        if args.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    # 6) gather timings to rank0 (best practice: aggregate across ranks)
    gathered = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered, times_ms)

    if rank == 0:
        # per-rank stats
        per_rank_mean = [stats.mean(ts) for ts in gathered]
        per_rank_std = [stats.pstdev(ts) if len(ts) > 1 else 0.0 for ts in gathered]

        # global aggregate over all samples from all ranks
        all_samples = [t for ts in gathered for t in ts]
        global_mean = stats.mean(all_samples)
        global_std = stats.pstdev(all_samples) if len(all_samples) > 1 else 0.0

        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        msg_bytes = args.numel * bytes_per_elem
        msg_mib = msg_bytes / (1024 * 1024)

        print("=== all_reduce single-node benchmark ===")
        print(f"backend={args.backend} device={args.device} world_size={args.world_size}")
        print(f"dtype={args.dtype} numel={args.numel} (~{msg_mib:.3f} MiB per rank tensor)")
        print(f"warmup={args.warmup} iters={args.iters}")
        print("")
        print(f"[GLOBAL] mean={global_mean:.3f} ms  std={global_std:.3f} ms  (over {len(all_samples)} samples)")
        for r in range(args.world_size):
            print(f"[rank {r}] mean={per_rank_mean[r]:.3f} ms  std={per_rank_std[r]:.3f} ms")

    cleanup_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--numel", type=int, default=1_000_000, help="Number of elements in the tensor to all-reduce.")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--master-addr", type=str, default="127.0.0.1")

    args = p.parse_args()

    # sanity: NCCL expects CUDA tensors
    if args.backend == "nccl" and args.device != "cuda":
        raise ValueError("backend=nccl requires device=cuda")
    if args.device == "cuda" and args.world_size > torch.cuda.device_count():
        raise ValueError(f"world_size={args.world_size} but only {torch.cuda.device_count()} CUDA devices available")

    master_port = _pick_free_port()

    mp.spawn(
        fn=benchmark_worker,
        args=(args, master_port),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
