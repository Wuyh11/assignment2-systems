import torch

def run_accumulation_test():
    print("=== Problem (mixed_precision_accumulation) Results ===")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 实验目标：累加 1000 次 0.01，理论结果应为 10.0

    # Case 1: FP32 累加器 + FP32 增量
    # 这是基准，精度最高
    s = torch.tensor(0, dtype=torch.float32)
    val_fp32 = torch.tensor(0.01, dtype=torch.float32)
    for i in range(1000):
        s += val_fp32
    print(f"\n1. FP32 accumulator + FP32 value:")
    print(f"   Result: {s.item():.6f} (Expected: 10.000000)")

    # Case 2: FP16 累加器 + FP16 增量
    # 问题：当 s 增大到一定程度（如 2048），FP16 的最小精度间隔会超过 0.01，
    # 导致后续的加法完全无效（"Underflow" 或 "Swallowing"）
    s = torch.tensor(0, dtype=torch.float16)
    val_fp16 = torch.tensor(0.01, dtype=torch.float16)
    for i in range(1000):
        s += val_fp16
    print(f"\n2. FP16 accumulator + FP16 value:")
    print(f"   Result: {s.item():.6f}")
    print("   Analysis: Result < 10.0 because small increments were lost due to low precision.")

    # Case 3: FP32 累加器 + FP16 增量 (隐式转换)
    # PyTorch 会自动将 fp16 提升为 fp32 进行加法，保留了累加器的精度
    s = torch.tensor(0, dtype=torch.float32)
    val_fp16 = torch.tensor(0.01, dtype=torch.float16)
    for i in range(1000):
        s += val_fp16
    print(f"\n3. FP32 accumulator + FP16 value (Implicit Cast):")
    print(f"   Result: {s.item():.6f}")

    # Case 4: FP32 累加器 + FP16 增量 (显式转换)
    # 模拟混合精度训练中的“权重更新”步骤：梯度是 FP16，但权重是 FP32
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(f"\n4. FP32 accumulator + FP16 value (Explicit Cast):")
    print(f"   Result: {s.item():.6f}")

if __name__ == "__main__":
    run_accumulation_test()