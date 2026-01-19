import torch
import time
import pandas as pd
from cs336_basics.model import scaled_dot_product_attention

def benchmark_attention():
    # 配置参数
    BATCH_SIZE = 8
    # 根据题目要求：Cartesian product of d_head and sequence_length
    D_HEADS = [16, 32, 64, 128]
    SEQ_LENS = [256, 1024, 4096, 8192, 16384]
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on {device}...\n")

    for d_head in D_HEADS:
        for seq_len in SEQ_LENS:
            print(f"Benchmarking: d_head={d_head}, seq_len={seq_len} ...", end=" ")
            
            try:
                # 1. 准备数据
                # 形状为 (Batch, Seq_Len, D_Head)，模拟单头注意力 (remove head dimension)
                # requires_grad=True 用于支持反向传播
                dtype = torch.float32
                q = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
                k = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
                v = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
                
                # 梯度输入，用于 backward
                grad_output = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, dtype=dtype)

                # 2. Warm-up (预热)
                # 运行几次 forward 和 backward 以消除启动开销
                for _ in range(5):
                    out = scaled_dot_product_attention(q, k, v)
                    out.backward(grad_output)
                    # 清除梯度，防止累积占用显存
                    q.grad = None
                    k.grad = None
                    v.grad = None
                
                torch.cuda.synchronize()
                
                # 3. 测量显存 (Memory Usage)
                # 题目要求：Measure how much memory is in use before the backward pass starts
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
                
                # 运行一次前向传播，建立计算图，此时激活值(Activations)存储在显存中
                out = scaled_dot_product_attention(q, k, v)
                
                # 获取当前分配的最大显存（即前向传播结束，反向传播开始前的状态）
                mem_bytes = torch.cuda.max_memory_allocated(device)
                mem_mb = mem_bytes / (1024 ** 2)
                
                # 清理这一次 forward 的结果，准备进行计时测试
                del out
                q.grad = None
                k.grad = None
                v.grad = None
                torch.cuda.empty_cache()

                # 4. 计时 Forward Pass (100 次)
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                for _ in range(100):
                    _ = scaled_dot_product_attention(q, k, v)
                
                torch.cuda.synchronize()
                fwd_latency = (time.perf_counter() - start_time) / 100 * 1000  # 转换为 ms

                # 5. 计时 Backward Pass (100 次)
                # 注意：为了单独测量 backward，我们需要在循环中每次都先运行 forward 来重建计算图
                # 我们只计时 .backward() 的部分
                bwd_times = []
                
                for _ in range(100):
                    # 前向传播 (不计入 backward 时间)
                    out = scaled_dot_product_attention(q, k, v)
                    
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    
                    # 反向传播
                    out.backward(grad_output)
                    
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    bwd_times.append(t1 - t0)
                    
                    # 清理梯度
                    q.grad = None
                    k.grad = None
                    v.grad = None

                bwd_latency = (sum(bwd_times) / len(bwd_times)) * 1000 # 转换为 ms

                print(f"Success. Mem: {mem_mb:.2f} MB, Fwd: {fwd_latency:.2f} ms, Bwd: {bwd_latency:.2f} ms")
                
                results.append({
                    "d_head": d_head,
                    "seq_len": seq_len,
                    "fwd_latency_ms": fwd_latency,
                    "bwd_latency_ms": bwd_latency,
                    "memory_mb": mem_mb
                })

            except torch.cuda.OutOfMemoryError:
                print("OOM (Out of Memory)")
                torch.cuda.empty_cache()
                results.append({
                    "d_head": d_head,
                    "seq_len": seq_len,
                    "fwd_latency_ms": "OOM",
                    "bwd_latency_ms": "OOM",
                    "memory_mb": "OOM"
                })
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    "d_head": d_head,
                    "seq_len": seq_len,
                    "fwd_latency_ms": "Error",
                    "bwd_latency_ms": "Error",
                    "memory_mb": "Error"
                })

    # 输出表格
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    benchmark_attention()