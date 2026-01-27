import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import math
from collections import deque

class DDPWithBucketedGradients:
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        """
        初始化一个分布式数据并行训练类，并设置桶化梯度通信。
        
        :param module: PyTorch 的 nn.Module 模型
        :param bucket_size_mb: 每个桶的大小（MB），用来控制每个梯度桶的最大大小。
        """
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.world_size = dist.get_world_size()  # 获取进程总数
        self.rank = dist.get_rank()  # 获取当前进程的 rank
        self.param_buckets = self._create_param_buckets(module)
        self._broadcast_weights()

    def _create_param_buckets(self, module):
        """
        将模型的参数分配到不同的桶中。
        
        :param module: 需要进行梯度桶化的 PyTorch 模型
        :return: 包含桶的列表，每个桶包含一组模型参数。
        """
        params = list(module.parameters())
        buckets = []
        current_bucket = []
        current_size = 0

        for param in params:
            param_size = param.numel() * param.element_size() / 1024 / 1024  # 参数的大小（MB）
            if current_size + param_size > self.bucket_size_mb:
                buckets.append(current_bucket)
                current_bucket = [param]
                current_size = param_size
            else:
                current_bucket.append(param)
                current_size += param_size

        if current_bucket:
            buckets.append(current_bucket)
        
        return buckets

    def _broadcast_weights(self):
        """
        广播模型的权重到所有进程，确保每个进程开始时都有相同的权重。
        """
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)  # 从 rank 0 广播到其他进程

    def forward(self, *inputs, **kwargs):
        """
        调用模型的 forward 方法进行前向传播。
        
        :param inputs: 前向传播的输入
        :param kwargs: 前向传播的额外参数
        :return: 模型的输出
        """
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        等待异步梯度通信完成。
        """
        dist.barrier()  # 等待所有进程同步完成

    def _overlap_computation_and_communication(self):
        """
        在反向传播过程中重叠计算和通信。
        """
        # 逐个桶进行梯度同步
        for bucket in self.param_buckets:
            for param in bucket:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)  # 梯度求和的 all-reduce 操作

    def train(self, data_loader, optimizer, num_epochs=10):
        """
        训练模型，并在每个 epoch 后调用 `finish_gradient_synchronization` 来确保通信同步。
        
        :param data_loader: 数据加载器
        :param optimizer: 优化器
        :param num_epochs: 训练的 epoch 数
        """
        self.module.train()
        for epoch in range(num_epochs):
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = nn.MSELoss()(outputs, labels)
                loss.backward()

                # 在反向传播过程中重叠计算和通信
                self._overlap_computation_and_communication()

                optimizer.step()
            
            # 确保所有梯度通信完成
            self.finish_gradient_synchronization()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
