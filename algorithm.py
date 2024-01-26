"""AdamW optimizer with warmup+cosine LR in PyTorch."""

from typing import Dict

from absl import logging
import torch
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from torch import nn

from pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]


def init_optimizer_state(n_steps,
                         model: nn.Module,
                         hyperparameters,
                         ):
  """Creates an AdamW optimizer and a learning rate schedule."""

  optimizer_state = {
      'optimizer':
          torch.optim.AdamW(
              model.parameters(),
              lr=hyperparameters.learning_rate,
              betas=(1.0 - hyperparameters.one_minus_beta1,
                     hyperparameters.beta2),
              eps=1e-8,
              weight_decay=hyperparameters.weight_decay,
              fused=False),
  }

  def pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
    warmup_steps = hyperparameters.warmup_factor * step_hint
    warmup = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps])

  optimizer_state['scheduler'] = pytorch_cosine_warmup(
      n_steps, hyperparameters, optimizer_state['optimizer'])

  return optimizer_state

