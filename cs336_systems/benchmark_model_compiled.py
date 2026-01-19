import torch
import timeit
import argparse
import numpy as np
import pandas as pd
from cs336_basics.model import BasicsTransformerLM

# 定义模型配置 (同 benchmark.py)
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def get_model(model_size, vocab_size, context_length, device):
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    )
    return model.to(device)

def run_timing_loop(model, x, optimizer, mode, steps, device):
    """辅助函数：运行计时循环"""
    times = []
    
    # 确定是否需要 backward 和 optimizer
    run_backward = mode in ["forward-backward", "train-step"]
    run_optimizer = mode == "train-step"

    if run_optimizer and optimizer is None:
        raise ValueError("Optimizer is required for train-step mode")

    for _ in range(steps):
        # 清零梯度 (如果需要)
        if run_backward:
            if run_optimizer:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        # Forward
        if not run_backward:
            with torch.no_grad():
                output = model(x)
        else:
            output = model(x)
            loss = output.mean() # Dummy loss
            
            # Backward
            loss.backward()
            
            # Optimizer Step
            if run_optimizer:
                optimizer.step()

        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    return np.mean(times), np.std(times)

def benchmark_end_to_end_compiled():
    # 固定参数用于对比
    context_lengths = [128, 256, 512, 1024] # 根据题目要求调整
    batch_size = 4
    vocab_size = 10000
    warmup_steps = 5
    measure_steps = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running End-to-End Compiled Benchmark on {device}")
    
    results = []

    # 遍历不同的模型大小
    # 注意：大型模型可能会 OOM，这里仅示例 medium，你可以通过命令行参数控制循环
    for model_size in ["medium"]: 
        for context_length in context_lengths:
            print(f"\nConfig: Size={model_size}, Context={context_length}")

            try:
                # 准备数据
                x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
                
                # ==========================
                # 1. Vanilla Model
                # ==========================
                print("  Running Vanilla...", end=" ")
                model_vanilla = get_model(model_size, vocab_size, context_length, device)
                optimizer_vanilla = torch.optim.AdamW(model_vanilla.parameters(), lr=1e-3)
                model_vanilla.train()

                # Warmup
                run_timing_loop(model_vanilla, x, optimizer_vanilla, "train-step", warmup_steps, device)
                # Benchmark Fwd Only
                fwd_vanilla, _ = run_timing_loop(model_vanilla, x, None, "forward-only", measure_steps, device)
                # Benchmark Train Step (Fwd+Bwd+Opt)
                step_vanilla, _ = run_timing_loop(model_vanilla, x, optimizer_vanilla, "train-step", measure_steps, device)
                print("Done.")

                # 清理显存
                del model_vanilla, optimizer_vanilla
                torch.cuda.empty_cache()

                # ==========================
                # 2. Compiled Model
                # ==========================
                print("  Running Compiled...", end=" ")
                model_compiled = get_model(model_size, vocab_size, context_length, device)
                optimizer_compiled = torch.optim.AdamW(model_compiled.parameters(), lr=1e-3)
                model_compiled.train()
                
                # 编译模型
                model_compiled = torch.compile(model_compiled)

                # Warmup (编译发生在这里)
                run_timing_loop(model_compiled, x, optimizer_compiled, "train-step", warmup_steps + 5, device) # 多预热几次
                # Benchmark Fwd Only
                fwd_compiled, _ = run_timing_loop(model_compiled, x, None, "forward-only", measure_steps, device)
                # Benchmark Train Step
                step_compiled, _ = run_timing_loop(model_compiled, x, optimizer_compiled, "train-step", measure_steps, device)
                print("Done.")

                # 记录结果
                results.append({
                    "model_size": model_size,
                    "context_len": context_length,
                    "vanilla_fwd_s": fwd_vanilla,
                    "compiled_fwd_s": fwd_compiled,
                    "vanilla_step_s": step_vanilla,
                    "compiled_step_s": step_compiled,
                    "speedup_step": step_vanilla / step_compiled
                })

                # 清理
                del model_compiled, optimizer_compiled
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print("OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error: {e}")

    # 输出对比表格
    if results:
        df = pd.DataFrame(results)
        print("\nEnd-to-End Results (Vanilla vs Compiled):")
        print(df.to_markdown(index=False))

if __name__ == "__main__":
    benchmark_end_to_end_compiled()