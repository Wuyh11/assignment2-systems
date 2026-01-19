import torch
import torch.nn as nn
import time
import argparse
import sys
import os
from contextlib import nullcontext

# ==========================================
# 导入您的自定义模型
# ==========================================
try:
    # 尝试直接导入
    from cs336_basics.model import BasicsTransformerLM
except ImportError:
    # 如果脚本放在 cs336-basics 目录下，尝试添加当前路径
    sys.path.append(os.getcwd())
    try:
        from cs336_basics.model import BasicsTransformerLM
    except ImportError:
        print("错误: 无法导入 'cs336_basics.model'。")
        print("请确保您在包含 'cs336_basics' 包的目录下运行此脚本，或设置了 PYTHONPATH。")
        sys.exit(1)

# ==========================================
# 配置参数 (参考 PDF Table 1)
# ==========================================
# 词表大小默认为 10,000 (PDF §1.1.2)
VOCAB_SIZE = 10000 

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def run_benchmark(model_size, batch_size, context_length, use_mixed_precision, device="cuda"):
    """
    运行基准测试的主函数
    """
    print(f"\n>>> Benchmarking Model: {model_size} | Context: {context_length} | Mixed Precision (BF16): {use_mixed_precision}")

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU. Results will not reflect GPU performance.")
        device = "cpu"

    # 1. 获取配置
    config = MODEL_CONFIGS[model_size]
    
    # 2. 初始化模型 (使用 cs336_basics 中的 BasicsTransformerLM)
    # 注意：您的模型 __init__ 需要以下参数
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0, # 默认 RoPE theta
    ).to(device)
    
    # 打印参数量以确认模型加载正确
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")

    # 3. 准备数据
    # BasicsTransformerLM 的 forward 只需要输入 token IDs (Int Tensor)
    inputs = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)
    
    # 4. 设置混合精度上下文
    # 作业要求使用 bfloat16
    dtype = torch.float16 if use_mixed_precision else torch.float32
    ctx = torch.autocast(device_type=device, dtype=dtype) if use_mixed_precision else nullcontext()

    loss_fn = nn.CrossEntropyLoss()
    # 虽然题目只测 forward/backward，但为了模拟真实训练，我们加上 optimizer step (不影响计时逻辑，只要计时段正确)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ==========================================
    # Warm-up (预热)
    # ==========================================
    print("Warming up (5 steps)...")
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits = model(inputs)
            # view(-1, vocab_size) 用于展平 batch 和 seq 维度以计算 loss
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # ==========================================
    # 计时 1: 仅前向传播 (Forward Only)
    # ==========================================
    steps = 10
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad(): # 纯 Forward 测试通常不需要梯度
        for _ in range(steps):
            with ctx:
                _ = model(inputs)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_fwd_time = start_event.elapsed_time(end_event) / steps
    print(f"Forward Pass Avg Time: {avg_fwd_time:.2f} ms")

    # ==========================================
    # 计时 2: 前向 + 反向 (Forward + Backward)
    # ==========================================
    start_event.record()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        # 注意：这里不包含 optimizer.step()，因为通常 backward 后就是梯度的产出点
    end_event.record()
    torch.cuda.synchronize()

    avg_fwd_bwd_time = start_event.elapsed_time(end_event) / steps
    print(f"Forward + Backward Avg Time: {avg_fwd_bwd_time:.2f} ms")

    return avg_fwd_time, avg_fwd_bwd_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark cs336_basics Transformer")
    parser.add_argument("--model_size", type=str, default="medium", choices=MODEL_CONFIGS.keys(), help="Model size from Table 1")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default 4)")
    parser.add_argument("--context_len", type=int, default=128, help="Context length")
    
    args = parser.parse_args()

    # 运行对比实验
    print(f"Running comparison for configuration: {args.model_size} model, {args.context_len} ctx len")
    
    # 1. FP32
    fwd_32, fb_32 = run_benchmark(
        args.model_size, args.batch_size, args.context_len, 
        use_mixed_precision=False
    )

    # 2. Mixed Precision (BF16)
    try:
        fwd_16, fb_16 = run_benchmark(
            args.model_size, args.batch_size, args.context_len, 
            use_mixed_precision=True
        )
        
        print("\n" + "="*40)
        print(f"SUMMARY RESULTS ({args.model_size}, ctx={args.context_len})")
        print("="*40)
        print(f"{'Metric':<20} | {'FP32 (ms)':<10} | {'BF16 (ms)':<10} | {'Speedup':<10}")
        print("-" * 60)
        print(f"{'Forward':<20} | {fwd_32:<10.2f} | {fwd_16:<10.2f} | {fwd_32/fwd_16:.2f}x")
        print(f"{'Fwd + Bwd':<20} | {fb_32:<10.2f} | {fb_16:<10.2f} | {fb_32/fb_16:.2f}x")
        print("="*40)
        
    except RuntimeError as e:
        print(f"\nRun failed for BF16: {e}")
        print("Ensure your GPU supports BFloat16 (Ampere+ architecture).")