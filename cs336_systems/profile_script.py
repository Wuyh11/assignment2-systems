import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
import argparse
import math
from einops import einsum
# 确保引用路径正确，根据你的项目结构调整
import cs336_basics.model 
from cs336_basics.model import BasicsTransformerLM, softmax

# -----------------------------------------------------------------------------
# 1. 定义带有 NVTX 标记的 Attention 函数 (对应作业 Source 116-143)
# -----------------------------------------------------------------------------
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """
    带有 NVTX 标记的 Attention 实现，用于 Nsight Systems 分析。
    """
    with nvtx.range("scaled dot product attention"):
        d_k = K.shape[-1]
        
        with nvtx.range("computing attention scores"):
            # QK^T / sqrt(d_k)
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx.range("computing softmax"):
            # Softmax
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx.range("final matmul"):
            # A * V
            output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
            
    return output

# 应用 Monkey Patch：替换原模型中的函数
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
print("Successfully patched scaled_dot_product_attention with NVTX annotations.")

# -----------------------------------------------------------------------------
# 2. 模型配置 (与 Table 1 一致)
# -----------------------------------------------------------------------------
MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def get_model(model_size, context_length, device):
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}")
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    )
    return model.to(device)

# -----------------------------------------------------------------------------
# 3. Profiling 主逻辑
# -----------------------------------------------------------------------------
def run_profile(model_size, context_length, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling Config: Size={model_size}, Context={context_length}, Batch={batch_size}")

    # 初始化
    model = get_model(model_size, context_length, device)
    model.train() # 必须是训练模式
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 数据
    vocab_size = 10000
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    # Warm-up (不记录 NVTX)
    print("Warm-up...")
    for _ in range(3):
        optimizer.zero_grad()
        out = model(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # 正式 Profiling Step
    # 使用 NVTX 标记整个步骤，方便在 Nsight 中快速定位
    print("Running profiling step...")
    torch.cuda.synchronize()
    
    with nvtx.range(f"Training Step - {model_size}"):
        
        optimizer.zero_grad()
        
        with nvtx.range("Forward Pass"):
            output = model(x)
            loss = output.mean()
        
        with nvtx.range("Backward Pass"):
            loss.backward()
        
        with nvtx.range("Optimizer Step"):
            optimizer.step()

    torch.cuda.synchronize()
    print("Profiling step finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="medium", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    run_profile(args.model_size, args.context_length, args.batch_size)