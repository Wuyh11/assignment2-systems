from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist


@dataclass
class _GradWork:
    param: torch.nn.Parameter
    handle: dist.Work  # distributed request handle (returned by async_op=True)


class DDP:
    """
    DDP wrapper that overlaps backward computation with communication by
    launching an async all-reduce for each parameter gradient as soon as it is ready.

    Public interface (as recommended in the PDF):
      - __init__(module)
      - forward(*inputs, **kwargs)
      - finish_gradient_synchronization()
    """

    def __init__(self, module: torch.nn.Module):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before constructing DDP.")

        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Stores in-flight async all-reduce handles for the *current* backward pass
        self._inflight: List[_GradWork] = []

        # Broadcast initial parameters so all ranks start from the same weights
        # (rank 0 is the source of truth).
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)

        # Register a post-accumulate-grad hook on each parameter.
        # This fires when param.grad has been accumulated for that param.
        for p in self.module.parameters():
            if not p.requires_grad:
                continue

            # We use a small state on the parameter to avoid scheduling multiple all-reduces
            # for the same param within a single backward (can happen in weird graphs or
            # if hooks trigger more than once).
            p._ddp_allreduce_scheduled = False  # type: ignore[attr-defined]

            def _make_hook(param: torch.nn.Parameter) -> Callable[[torch.Tensor], None]:
                def _hook(grad: torch.Tensor) -> None:
                    # grad is param.grad (post-accumulation)
                    if grad is None:
                        return
                    # Don't schedule twice for the same backward pass.
                    if getattr(param, "_ddp_allreduce_scheduled", False):
                        return
                    setattr(param, "_ddp_allreduce_scheduled", True)

                    # Launch async all-reduce on the gradient tensor (in-place)
                    # NOTE: we do NOT divide by world_size here to keep it overlappable;
                    # we do it after wait() in finish_gradient_synchronization().
                    handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                    self._inflight.append(_GradWork(param=param, handle=handle))

                return _hook

            p.register_post_accumulate_grad_hook(_make_hook(p))

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """
        Wait for all async all-reduce ops launched during backward to be queued/completed
        so grads are ready for optimizer.step().
        Also performs the average (divide by world_size).
        """
        # Wait for each all-reduce to complete, then average the gradient.
        # (The PDF recommends calling this right before optimizer.step()).
        for work in self._inflight:
            work.handle.wait()
            if work.param.grad is not None:
                work.param.grad.div_(self.world_size)

            # Reset scheduled flag for the next iteration/backward.
            setattr(work.param, "_ddp_allreduce_scheduled", False)

        self._inflight.clear()
