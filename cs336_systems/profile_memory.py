import torch
import torch.nn as nn
import argparse
import sys
import os
import time
from contextlib import nullcontext
import pickle # 用于手动保存 snapshot (如果 _dump_snapshot 也不可用)

# ==========================================
# 导入模型
# ==========================================
try:
    from cs336_basics.model import BasicsTransformerLM
except ImportError:
    sys.path.append(os.getcwd())
    try:
        from cs336_basics.model import BasicsTransformerLM
    except ImportError:
        print("错误: 无法导入 'cs336_basics.model'。请在项目根目录下运行此脚本。")
        sys.exit(1)

# ==========================================
# 模型配置 (Medium)
# ==========================================
MODEL_CONFIGS = {
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
}

VOCAB_SIZE = 10000

def run_memory_profiling(model_size, batch_size, context_len, mode, use_amp):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Error: Memory profiling requires a GPU.")
        return

    print(f"\n=== Memory Profiling: {model_size.upper()} | Mode: {mode} | Context: {context_len} | AMP: {use_amp} ===")
    
    # 1. 初始化模型
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_len,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    # 2. 准备组件
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    inputs = torch.randint(0, VOCAB_SIZE, (batch_size, context_len), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (batch_size, context_len), device=device)

    dtype = torch.float16 if use_amp else torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()
    
    scaler = torch.cuda.amp.GradScaler() if (use_amp and mode == "training") else None

    # ==========================================
    # Warm-up (预热)
    # ==========================================
    print("Warming up...")
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits = model(inputs)
            if mode == "training":
                loss = loss_fn(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        
        if mode == "training":
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
    torch.cuda.synchronize()

    # ==========================================
    # 核心：Memory Profiling
    # ==========================================
    print("Starting memory recording...")
    
    # [Fix 1] 使用带下划线的 API
    try:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    except AttributeError:
         print("Warning: torch.cuda.memory._record_memory_history not found. Trying without underscore.")
         torch.cuda.memory.record_memory_history(max_entries=1000000)

    try:
        optimizer.zero_grad(set_to_none=True)
        
        # A. Forward
        with ctx:
            logits = model(inputs)
            if mode == "training":
                loss = loss_fn(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        
        # B. Backward + Step
        if mode == "training":
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize()

        # [Fix 2] 使用带下划线的 API，并增加兼容性处理
        filename = f"memory_{model_size}_{mode}_ctx{context_len}_amp{int(use_amp)}.pickle"
        
        try:
            # 尝试标准实验性 API (2.1+)
            torch.cuda.memory._dump_snapshot(filename)
        except AttributeError:
            try:
                # 尝试旧版/无下划线版
                torch.cuda.memory_dump_snapshot(filename)
            except AttributeError:
                # 如果都没有，尝试获取 snapshot 字典并手动保存
                print("Falling back to manual pickle dump...")
                snapshot = torch.cuda.memory_snapshot()
                with open(filename, 'wb') as f:
                    pickle.dump(snapshot, f)
        
        print(f"Snapshot saved to: {filename}")
        print(f"-> Drag this file to https://pytorch.org/memory_viz to view.")

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak Memory Allocated: {peak_mem:.2f} MB")

    except torch.cuda.OutOfMemoryError:
        print("OOM: Out Of Memory! Try reducing context_len or batch_size.")
    
    finally:
        # [Fix 3] 停止录制
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except AttributeError:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="medium", choices=["small", "medium"])
    parser.add_argument("--mode", type=str, default="training", choices=["inference", "training"])
    parser.add_argument("--context_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    
    args = parser.parse_args()

    run_memory_profiling(
        args.model_size, 
        args.batch_size, 
        args.context_len, 
        args.mode, 
        args.mixed_precision
    )