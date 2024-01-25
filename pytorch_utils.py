import os
from typing import Tuple
from absl import logging
import torch
from torch import nn
import torch.distributed as dist
import gc
from profiler import Profiler
import time



def pytorch_setup() -> Tuple[bool, int, torch.device, int]:
  use_pytorch_ddp = 'LOCAL_RANK' in os.environ
  rank = int(os.environ['LOCAL_RANK']) if use_pytorch_ddp else 0
  device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
  n_gpus = torch.cuda.device_count()
  return use_pytorch_ddp, rank, device, n_gpus


def pytorch_init(use_pytorch_ddp: bool, rank: int, profiler: Profiler) -> None:
  # From the docs: "(...) causes cuDNN to benchmark multiple convolution
  # algorithms and select the fastest."
  torch.backends.cudnn.benchmark = True

  if use_pytorch_ddp:
    torch.cuda.set_device(rank)
    profiler.set_local_rank(rank)
    # Only log once (for local rank == 0).
    if rank != 0:

      def logging_pass(*args):
        pass

      logging.info = logging_pass
    # Initialize the process group.
    dist.init_process_group('nccl')


def sync_ddp_time(time: float, device: torch.device) -> float:
  time_tensor = torch.tensor(time, dtype=torch.float64, device=device)
  dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
  return time_tensor.item()


def update_batch_norm_fn(module: nn.Module,
                         update_batch_norm: bool) -> None:
  bn_layers = (
      torch.nn.modules.batchnorm._BatchNorm,  # PyTorch BN base class.
  )
  if isinstance(module, bn_layers):
    if not update_batch_norm:
      module.eval()
      module.momentum_backup = module.momentum
      # module.momentum can be float or torch.Tensor.
      module.momentum = 0. * module.momentum_backup
    elif hasattr(module, 'momentum_backup'):
      module.momentum = module.momentum_backup
    module.track_running_stats = update_batch_norm

def _get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def _get_time_ddp():
    torch.cuda.synchronize()
    t = time.time()
    return sync_ddp_time(t, DEVICE)

def _reset_cuda_mem():
    if torch.cuda.is_available():
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()