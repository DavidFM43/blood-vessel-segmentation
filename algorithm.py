"""AdamW optimizer with warmup+cosine LR in PyTorch."""

from typing import Dict

from absl import logging
import torch
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from torch import nn
import workload
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

def update_params(model,
                  hyperparameters,
                  batch: Dict[str, torch.Tensor],
                  optimizer_state,
                  global_step: int,
                  rng,
                  metric_logger = None
                  ):
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  model.train()
  optimizer_state['optimizer'].zero_grad()

  logits_batch = workload.model_fn(
      model=model,
      batch=batch,
      mode="train",
      rng=rng,
      update_batch_norm=True)

  loss = workload.loss_fn(
      targets=batch['targets'],
      inputs=logits_batch
  )
  loss.backward()

  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
    torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=grad_clip)
  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 10 or global_step % 500 == 0:
    with torch.no_grad():
      parameters = [p for p in model.parameters() if p.grad is not None]
      grad_norm = torch.norm(
          torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
    if metrics_logger is not None:
      workload.metrics_logger.append_scalar_metrics(
          {
              'loss': loss.item(),
              'grad_norm': grad_norm.item(),
          }, global_step)
    logging.info(f'{global_step}) loss = {loss.item():0.3f}, grad_norm = {grad_norm.item():0.3f}'),

  return optimizer_state, model