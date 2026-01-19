import torch
import timeit
import argparse
import numpy as np
import pandas as pd
from cs336_basics.model import BasicsTransformerLM

# 根据作业文档 Table 1 定义的模型配置
MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def get_model(model_size, vocab_size, context_length, device):
    """根据配置初始化模型"""
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_size]
    # 初始化模型
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

def benchmark(
    model_size: str,
    context_length: int,
    batch_size: int,
    mode: str,
    warmup_steps: int,
    num_steps: int
):
    """
    执行基准测试的核心函数。
    返回 (mean_time, std_time)，如果发生 OOM 则返回 (None, None)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 清理缓存，防止之前的测试影响
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"Benchmarking: Size={model_size}, Ctx={context_length}, Mode={mode}, Warmup={warmup_steps} ... ", end="", flush=True)

    try:
        # 1. 初始化模型
        vocab_size = 10000 
        model = get_model(model_size, vocab_size, context_length, device)
        
        if mode == "forward-backward":
            model.train()
        else:
            model.eval()

        # 2. 生成随机数据
        x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

        # 3. 预热 (Warm-up)
        if warmup_steps > 0:
            for _ in range(warmup_steps):
                model.zero_grad()
                if mode == "forward-only":
                    with torch.no_grad():
                        _ = model(x)
                elif mode == "forward-backward":
                    output = model(x)
                    loss = output.mean() 
                    loss.backward()
                torch.cuda.synchronize()

        # 4. 计时循环
        times = []
        for _ in range(num_steps):
            model.zero_grad()
            
            torch.cuda.synchronize()
            start_time = timeit.default_timer()

            if mode == "forward-only":
                with torch.no_grad():
                    _ = model(x)
            elif mode == "forward-backward":
                output = model(x)
                loss = output.mean()
                loss.backward()

            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            
            times.append(end_time - start_time)

        # 5. 统计
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Done. ({mean_time:.4f}s)")
        
        # 删除模型和数据以释放显存
        del model
        del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return mean_time, std_time

    except torch.cuda.OutOfMemoryError:
        print("FAILED (OOM)")
        # 尝试清理以便后续测试能继续
        if 'model' in locals(): del model
        if 'x' in locals(): del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None
    except Exception as e:
        print(f"FAILED ({e})")
        return None, None

def run_all_benchmarks(args):
    """
    自动运行作业要求的 (b) 和 (c) 部分测试，并生成表格。
    """
    results_b = []
    
    # ---------------------------------------------------------
    # Part (b): 不同模型大小的 Forward vs Backward (Warmup=5)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Running Part (b): Model Sizing Benchmark")
    print("="*50)
    
    # 注意：如果 XL 或 2.7B 仍然 OOM，可以适当减小这里的 context_length
    ctx_len = args.context_length 
    bs = args.batch_size
    
    for size in MODEL_CONFIGS.keys():
        # 测 Forward
        fwd_mean, fwd_std = benchmark(size, ctx_len, bs, "forward-only", warmup_steps=5, num_steps=10)
        
        # 测 Forward + Backward
        fb_mean, fb_std = benchmark(size, ctx_len, bs, "forward-backward", warmup_steps=5, num_steps=10)
        
        # 计算 Backward (Fwd+Bwd - Fwd)
        if fwd_mean is not None and fb_mean is not None:
            bwd_mean = fb_mean - fwd_mean
        else:
            bwd_mean = None

        results_b.append({
            "Model Size": size,
            "Fwd (s)": fwd_mean,
            "Fwd Std": fwd_std,
            "Fwd+Bwd (s)": fb_mean,
            "Fwd+Bwd Std": fb_std,
            "Est. Bwd (s)": bwd_mean
        })

    # 输出 Part (b) 表格
    df_b = pd.DataFrame(results_b)
    print("\n--- Part (b) Results ---")
    print(df_b.to_markdown(index=False, floatfmt=".4f"))

    # ---------------------------------------------------------
    # Part (c): 预热步数的影响 (Warmup Analysis)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Running Part (c): Warm-up Analysis (using 'medium' model)")
    print("="*50)
    
    results_c = []
    target_model = "medium" # 使用 medium 进行测试，避免大模型 OOM 或小模型太快看不出区别
    
    warmup_settings = [0, 1, 2, 5]
    
    for w in warmup_settings:
        mean_t, std_t = benchmark(target_model, ctx_len, bs, "forward-backward", warmup_steps=w, num_steps=10)
        results_c.append({
            "Warmup Steps": w,
            "Time (s)": mean_t,
            "Std Dev": std_t
        })
        
    # 输出 Part (c) 表格
    df_c = pd.DataFrame(results_c)
    print("\n--- Part (c) Results ---")
    print(df_c.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking script for Transformer LM")
    
    parser.add_argument("--auto", action="store_true", help="Automatically run all tests for the assignment report")
    
    parser.add_argument("--model_size", type=str, default="medium", 
                        choices=list(MODEL_CONFIGS.keys()), help="Model size config")
    parser.add_argument("--context_length", type=int, default=128, help="Context length (default 128 to be safe)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--mode", type=str, default="forward-only", 
                        choices=["forward-only", "forward-backward"], help="Benchmarking mode")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps")
    parser.add_argument("--steps", type=int, default=10, help="Number of measurement steps")

    args = parser.parse_args()

    if args.auto:
        run_all_benchmarks(args)
    else:
        # 单次运行模式
        mean, std = benchmark(
            model_size=args.model_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            mode=args.mode,
            warmup_steps=args.warmup_steps,
            num_steps=args.steps
        )
        print(f"\nFinal Result: {mean:.6f} s ± {std:.6f} s")