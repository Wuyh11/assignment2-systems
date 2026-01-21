import torch
import triton
import triton.testing
import math
import csv
import os
from cs336_systems.flash_attention2_triton import FlashAttention2ForwardTriton

# -----------------------------------------------------------------------------
# 1. 实现 Baseline 和 Wrapper
# -----------------------------------------------------------------------------

def manual_attention(q, k, v, is_causal=False):
    """PyTorch 原生实现，用于对比 Baseline"""
    d = q.shape[-1]
    # S = QK^T / sqrt(d)
    S = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    
    if is_causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, v)

def triton_flash_attention(q, k, v, is_causal=False):
    """调用你的 Triton FlashAttention 实现"""
    return FlashAttention2ForwardTriton.apply(q, k, v, is_causal)

# -----------------------------------------------------------------------------
# 2. Benchmarking 核心逻辑
# -----------------------------------------------------------------------------

def run_benchmark(seq_len, d_model, dtype, provider, mode, batch_size=1, is_causal=True):
    # 初始化输入数据
    # 根据 Assignment 要求，Batch Size = 1
    # n_heads 设为 16 (你可以根据需要调整，Assignment Leaderboard 要求是 16)
    n_heads = 16
    
    try:
        q = torch.randn((batch_size, n_heads, seq_len, d_model), device='cuda', dtype=dtype, requires_grad=True)
        k = torch.randn((batch_size, n_heads, seq_len, d_model), device='cuda', dtype=dtype, requires_grad=True)
        v = torch.randn((batch_size, n_heads, seq_len, d_model), device='cuda', dtype=dtype, requires_grad=True)
    except torch.cuda.OutOfMemoryError:
        return float('nan')

    # 选择要测试的函数
    if provider == 'torch':
        fn = lambda: manual_attention(q, k, v, is_causal)
    elif provider == 'triton':
        fn = lambda: triton_flash_attention(q, k, v, is_causal)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # 预热并测量时间
    try:
        if mode == 'fwd':
            # 仅测量前向
            ms = triton.testing.do_bench(fn)
        
        elif mode == 'bwd':
            # 测量反向 (需要先跑一次前向生成 graph)
            o = fn()
            do = torch.randn_like(o)
            # retain_graph=True 不是必须的，除非我们需要多次反向，但 do_bench 内部会多次运行
            # 为了简单，定义一个只跑 backward 的闭包
            def bwd_fn():
                o.backward(do, retain_graph=True) # retain_graph 必须为 True 因为 do_bench 会循环调用
            
            # grad_to_none 确保梯度被重置，防止累积占用显存或影响计算
            ms = triton.testing.do_bench(bwd_fn, grad_to_none=[q, k, v])
            
        elif mode == 'fwd_bwd':
            # 测量端到端
            def fwd_bwd_fn():
                o = fn()
                do = torch.randn_like(o)
                o.backward(do)
            
            ms = triton.testing.do_bench(fwd_bwd_fn, grad_to_none=[q, k, v])
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    except torch.cuda.OutOfMemoryError:
        return float('nan')
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return float('nan')

    return ms

# -----------------------------------------------------------------------------
# 3. 主程序：循环遍历配置并保存 CSV
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # 配置扫描范围 (Assignment 要求的笛卡尔积)
    # 注意：根据显存大小，过大的 seq_len (如 16384) 在 PyTorch 原生实现下可能会 OOM
    seq_lens = [128, 256, 1024, 2048, 4096, 8192] # 可以尝试添加 16384
    d_models = [16, 32, 64, 128]
    dtypes = [torch.float32, torch.bfloat16]
    providers = ['torch', 'triton']
    modes = ['fwd', 'bwd', 'fwd_bwd']
    
    output_file = "flash_benchmark_results.csv"
    
    print(f"Starting benchmark... Saving to {output_file}")
    
    # 打印表头到控制台
    header = ["SeqLen", "D_Model", "Dtype", "Provider", "Mode", "Time(ms)"]
    print(f"{header[0]:<8} | {header[1]:<7} | {header[2]:<15} | {header[3]:<8} | {header[4]:<8} | {header[5]:<10}")
    print("-" * 75)

    # 打开 CSV 文件准备写入
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header) # 写入 CSV 表头

        for seq_len in seq_lens:
            for d_model in d_models:
                for dtype in dtypes:
                    for provider in providers:
                        for mode in modes:
                            
                            # 运行测试
                            ms = run_benchmark(
                                seq_len=seq_len, 
                                d_model=d_model, 
                                dtype=dtype, 
                                provider=provider, 
                                mode=mode
                            )
                            
                            # 格式化输出
                            dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
                            time_str = f"{ms:.4f}" if not math.isnan(ms) else "OOM/Error"
                            
                            # 打印到控制台
                            print(f"{seq_len:<8} | {d_model:<7} | {dtype_str:<15} | {provider:<8} | {mode:<8} | {time_str:<10}")
                            
                            # 写入 CSV
                            writer.writerow([seq_len, d_model, dtype_str, provider, mode, ms])
                            
                            # 立即刷新缓冲区，防止程序崩溃数据丢失
                            f.flush()

    print(f"\nBenchmark finished. Results saved to {output_file}")