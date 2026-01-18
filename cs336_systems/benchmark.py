import torch
import timeit
import argparse
import numpy as np
from cs336_basics.model import BasicsTransformerLM

# 根据作业文档 Table 1 定义的模型配置
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
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
        rope_theta=10000.0,  # 默认值
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
    执行基准测试的主函数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on {device}")
    print(f"Config: Size={model_size}, Context={context_length}, Batch={batch_size}, Mode={mode}")

    # 1. 初始化模型
    vocab_size = 10000  # 作业文档指定
    model = get_model(model_size, vocab_size, context_length, device)

    # 设置为训练模式（对于 backward 测试很重要）或评估模式
    if mode == "forward-backward":
        model.train()
    else:
        model.eval()

    # 2. 生成随机数据
    # 输入形状: (batch_size, context_length)
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    # 3. 预热 (Warm-up)
    print(f"Running {warmup_steps} warm-up steps...")
    for _ in range(warmup_steps):
        model.zero_grad()
        if mode == "forward-only":
            with torch.no_grad():
                _ = model(x)
        elif mode == "forward-backward":
            output = model(x)
            loss = output.mean()  # 简单的 dummy loss
            loss.backward()
        torch.cuda.synchronize()

    # 4. 计时循环 (Timing)
    print(f"Timing {num_steps} steps...")
    times = []

    for _ in range(num_steps):
        model.zero_grad()

        # 开始计时
        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        if mode == "forward-only":
            with torch.no_grad():
                _ = model(x)
        elif mode == "forward-backward":
            output = model(x)
            loss = output.mean()
            loss.backward()

        # 结束计时
        torch.cuda.synchronize()
        end_time = timeit.default_timer()

        times.append(end_time - start_time)

    # 5. 计算统计结果
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nResults for {model_size} (Context {context_length}):")
    print(f"Average time per step: {mean_time:.6f} s")
    print(f"Standard deviation:    {std_time:.6f} s")

    return mean_time, std_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking script for Transformer LM")

    parser.add_argument("--model_size", type=str, default="medium",
                        choices=list(MODEL_CONFIGS.keys()), help="Model size config")
    parser.add_argument("--context_length", type=int, default=128, help="Context length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--mode", type=str, default="forward-only",
                        choices=["forward-only", "forward-backward"], help="Benchmarking mode")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps")
    parser.add_argument("--steps", type=int, default=10, help="Number of measurement steps")

    args = parser.parse_args()

    benchmark(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        mode=args.mode,
        warmup_steps=args.warmup_steps,
        num_steps=args.steps
    )