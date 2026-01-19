import torch
import time
import pandas as pd
from cs336_basics.model import scaled_dot_product_attention

def benchmark_attention_compiled():
    # 配置参数
    BATCH_SIZE = 8
    D_HEADS = [16, 32, 64, 128]
    SEQ_LENS = [256, 1024, 4096, 8192, 16384]
    
    # 定义编译版本的函数
    # fullgraph=True 通常对纯 PyTorch 函数更有效，但如果遇到动态控制流问题可以去掉
    compiled_sdpa = torch.compile(scaled_dot_product_attention)
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running compiled attention benchmark on {device}...\n")

    for d_head in D_HEADS:
        for seq_len in SEQ_LENS:
            print(f"Benchmarking: d_head={d_head}, seq_len={seq_len} ...")
            
            try:
                dtype = torch.float32
                # 构造输入
                make_inputs = lambda: (
                    torch.randn(BATCH_SIZE, seq_len, d_head, device=device, dtype=dtype, requires_grad=True),
                    torch.randn(BATCH_SIZE, seq_len, d_head, device=device, dtype=dtype, requires_grad=True),
                    torch.randn(BATCH_SIZE, seq_len, d_head, device=device, dtype=dtype, requires_grad=True),
                    torch.randn(BATCH_SIZE, seq_len, d_head, device=device, dtype=dtype) # grad_output
                )

                # ==========================================
                # 1. 测试 Vanilla (原生) PyTorch
                # ==========================================
                q, k, v, grad_out = make_inputs()
                
                # Warmup Vanilla
                for _ in range(5):
                    out = scaled_dot_product_attention(q, k, v)
                    out.backward(grad_out)
                    q.grad = None; k.grad = None; v.grad = None
                torch.cuda.synchronize()

                # Timing Vanilla Forward
                start = time.perf_counter()
                for _ in range(100):
                    _ = scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                vanilla_fwd_ms = (time.perf_counter() - start) / 100 * 1000

                # Timing Vanilla Backward
                bwd_times = []
                for _ in range(100):
                    out = scaled_dot_product_attention(q, k, v)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out.backward(grad_out)
                    torch.cuda.synchronize()
                    bwd_times.append(time.perf_counter() - t0)
                    q.grad = None; k.grad = None; v.grad = None
                vanilla_bwd_ms = (sum(bwd_times) / len(bwd_times)) * 1000

                # ==========================================
                # 2. 测试 Compiled (编译) PyTorch
                # ==========================================
                # 清理显存以防干扰
                del q, k, v, grad_out, out
                torch.cuda.empty_cache()
                
                q, k, v, grad_out = make_inputs()

                # Warmup Compiled (非常重要，第一次运行会触发编译)
                # 编译通常需要比普通 warmup 更多的次数来稳定
                for _ in range(10): 
                    out = compiled_sdpa(q, k, v)
                    out.backward(grad_out)
                    q.grad = None; k.grad = None; v.grad = None
                torch.cuda.synchronize()

                # Timing Compiled Forward
                start = time.perf_counter()
                for _ in range(100):
                    _ = compiled_sdpa(q, k, v)
                torch.cuda.synchronize()
                compiled_fwd_ms = (time.perf_counter() - start) / 100 * 1000

                # Timing Compiled Backward
                bwd_times = []
                for _ in range(100):
                    out = compiled_sdpa(q, k, v)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out.backward(grad_out)
                    torch.cuda.synchronize()
                    bwd_times.append(time.perf_counter() - t0)
                    q.grad = None; k.grad = None; v.grad = None
                compiled_bwd_ms = (sum(bwd_times) / len(bwd_times)) * 1000

                # 记录结果
                results.append({
                    "d_head": d_head,
                    "seq_len": seq_len,
                    "vanilla_fwd_ms": vanilla_fwd_ms,
                    "compiled_fwd_ms": compiled_fwd_ms,
                    "vanilla_bwd_ms": vanilla_bwd_ms,
                    "compiled_bwd_ms": compiled_bwd_ms,
                    "speedup_fwd": vanilla_fwd_ms / compiled_fwd_ms,
                    "speedup_bwd": vanilla_bwd_ms / compiled_bwd_ms
                })
                
                print(f"  -> Vanilla Fwd: {vanilla_fwd_ms:.2f}ms, Compiled Fwd: {compiled_fwd_ms:.2f}ms")

            except torch.cuda.OutOfMemoryError:
                print("  -> OOM")
                torch.cuda.empty_cache()
                results.append({
                    "d_head": d_head, "seq_len": seq_len,
                    "vanilla_fwd_ms": "OOM", "compiled_fwd_ms": "OOM",
                    "vanilla_bwd_ms": "OOM", "compiled_bwd_ms": "OOM",
                    "speedup_fwd": 0, "speedup_bwd": 0
                })
            except Exception as e:
                print(f"  -> Error: {e}")

    # 输出表格
    df = pd.DataFrame(results)
    print("\nBenchmark Results (Vanilla vs Compiled):")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    benchmark_attention_compiled()