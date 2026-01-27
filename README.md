# CS336 Spring 2025 Assignment 2: Systems Optimization for Deep Learning

This project focuses on implementing and benchmarking various system-level optimizations for Transformer-based language models. As a part of the CS336 curriculum, I have implemented high-performance components including mixed-precision training, flash attention kernels, and distributed training strategies.

## Project Structure

- **[`cs336_systems/`](./cs336_systems/)**: This is the heart of the project where advanced systems features are implemented.
  - `mixed_precision_accumulation.py`: Implements numerical stability-aware gradient accumulation for mixed precision training.
  - `flash_attention2_triton.py` & `flash_attention2_pytorch.py`: High-performance Flash Attention v2 implementations using Triton and PyTorch's native APIs.
  - `ddp_naive.py` & `ddp_overlap_individual_parameters.py`: Implementations of Distributed Data Parallel (DDP). The overlapped version improves efficiency by synchronizing gradients while computing.
  - `benchmark_*.py`: Scripts to measure the performance gains of each optimization (e.g., memory usage, throughput).
  - `profile_*.py`: Scripts used for profiling execution via Nsight Systems to identify bottlenecks.
- **[`cs336-basics/`](./cs336-basics/)**: Contains the foundational Transformer architecture, optimizers, and data loading logic carried over from Assignment 1.
- **[`tests/`](./tests/)**: Unit and integration tests to ensure the correctness of DDP synchronization, sharded optimizers, and attention implementations.
- **[`test_data/`](./test_data/)**: Stores benchmark outputs (`.csv`) and profiling reports (`.nsys-rep`).

## Key Features
- **Mixed Precision**: Accelerated training using FP16/BF16 without loss of accuracy.
- **Compilation**: Leveraged `torch.compile` for kernel fusion and graph optimization.
- **Manual Kernels**: Custom Flash Attention kernels implemented in Triton for peak GPU utilization.
- **Distributed Training**: All-reduce based gradient synchronization with communication-computation overlap.

---

# CS336 Spring 2025 实验二：深度学习系统优化

本项目专注于为基于 Transformer 的语言模型实现并测试各种系统级优化。作为 CS336 课程的一部分，我实现了包括混合精度训练、Flash Attention 算子以及分布式训练策略在内的高性能组件。

## 项目结构

- **[`cs336_systems/`](./cs336_systems/)**: 项目的核心文件夹，包含了所有进阶系统特性的实现。
  - `mixed_precision_accumulation.py`: 实现了混合精度训练中的梯度累加，确保数值稳定性。
  - `flash_attention2_triton.py` & `flash_attention2_pytorch.py`: 分别使用 Triton 和 PyTorch 原生 API 实现的高性能 Flash Attention v2。
  - `ddp_naive.py` & `ddp_overlap_individual_parameters.py`: 分布式数据并行（DDP）的实现。重叠版本通过在计算时同步梯度来提升效率。
  - `benchmark_*.py`: 用于评估各项优化（如显存占用、吞吐量）性能提升的脚本。
  - `profile_*.py`: 使用 Nsight Systems 进行性能分析的脚本，用于定位系统瓶颈。
- **[`cs336-basics/`](./cs336-basics/)**: 包含从作业 1 延续下来的基础 Transformer 架构、优化器和数据加载逻辑。
- **[`tests/`](./tests/)**: 单元测试和集成测试，确保 DDP 同步、分片优化器和 Attention 实现的正确性。
- **[`test_data/`](./test_data/)**: 存储基准测试结果 (`.csv`) 和分析报告 (`.nsys-rep`)。

## 核心亮点
- **混合精度**: 使用 FP16/BF16 加速训练，同时保持模型精度。
- **模型编译**: 利用 `torch.compile` 进行算子融合和计算图优化。
- **自定义算子**: 在 Triton 中实现自定义 Flash Attention 内核，以达到极高的 GPU 利用率。
- **分布式训练**: 基于 All-reduce 的梯度同步，支持计算与通信的重叠（Overlap）。

## Setup

We use `uv` to manage dependencies. You can verify the installation by running:

```sh
$ uv run python
```

## Submitting

To submit, run `./test_and_make_submission.sh`. This script will install dependencies, run tests, and create a submission package.

