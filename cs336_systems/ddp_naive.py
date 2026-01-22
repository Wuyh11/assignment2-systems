import torch
import torch.distributed as dist


class NaiveDDP(torch.nn.Module):
    """
    Naïve Distributed Data Parallel:
    - init 时广播参数，保证所有 rank 初始权重一致
    - backward 之后手动同步梯度：逐参数 all_reduce + /world_size
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized (call dist.init_process_group first).")

        self.world_size = dist.get_world_size()

        # 让所有 rank 初始参数一致：从 rank0 广播
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def sync_gradients(self):
        """
        在 loss.backward() 之后调用：
        对每个参数的梯度做 all_reduce(sum)，再除以 world_size 得到 mean。
        """
        for p in self.module.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world_size)
