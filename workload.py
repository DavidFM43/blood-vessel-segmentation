from tqdm import tqdm
import pytorch_utils
import data_utils
from data_utils import KidneyDataset
import albumentations as A
import torch
import segmentation_models_pytorch as smp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import contextlib
from absl import logging
from torch import nn
import random_utils as prng
from surface_dice import SurfaceDiceMetric
from typing import Dict
from patcher import Patcher


USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()
patch_size = 224
patch_overlap = 50
enconder_name = "timm-mobilenetv3_small_075"
enconder_weights = None
in_channels = 1
num_classes = 1
loss_fn = nn.BCEWithLogitsLoss(reduction="none")
max_allowed_runtime_sec = 3600  # 1 hour
eval_period_time_sec = 600  # 10 min
PR_THRESHOLD = 0.5
num_workers = 2



def build_input_queue(split, data_dir, global_batch_size, rng=None, cycle=True, prefetch=True):
    is_train = split == "train"
    train_transforms = A.Compose([A.RandomCrop(patch_size, patch_size)])
    eval_transforms = None
    transforms = train_transforms if is_train else eval_transforms

    if rng is not None:
        torch.random.manual_seed(rng[0])

    ds = KidneyDataset(data_dir=data_dir, split=split, transforms=transforms)

    sampler = None
    if USE_PYTORCH_DDP:
        per_device_batch_size = global_batch_size // N_GPUS
        ds_iter_batch_size = per_device_batch_size
    else:
        ds_iter_batch_size = global_batch_size

    if USE_PYTORCH_DDP:
        if is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, num_replicas=N_GPUS, rank=RANK, shuffle=True
            )
        else:
            sampler = data_utils.DistributedEvalSampler(
                ds, num_replicas=N_GPUS, rank=RANK, shuffle=False
            )

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=ds_iter_batch_size,
        shuffle=not USE_PYTORCH_DDP and is_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=is_train,
    )
    if prefetch:
        dataloader = data_utils.PrefetchedWrapper(dataloader, DEVICE)
    if cycle:
        dataloader = data_utils.cycle(
            dataloader, custom_sampler=USE_PYTORCH_DDP, use_mixup=False
        )

    return dataloader


def init_model_fn(rng):
    torch.random.manual_seed(rng[0])
    model = smp.Unet(
        encoder_name=enconder_name,
        encoder_weights=enconder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    model.to(DEVICE)
    if N_GPUS > 1:
        if USE_PYTORCH_DDP:
            model = DDP(model, device_ids=[RANK], output_device=RANK)
        else:
            model = torch.nn.DataParallel(model)
    return model


def model_fn(model, batch, mode, update_batch_norm):
    del update_batch_norm
    inputs = batch["inputs"]
    bs, c, h, w =  inputs.shape
    patcher = Patcher(h, w, patch_size=patch_size, overlap=patch_overlap)

    if mode == "eval_train":
        mode = "eval"
    if mode == "eval":
        model.eval()
        inputs = patcher.extract_patches(inputs)  # (B, n_patches, C, H, W)
        inputs = inputs.flatten(end_dim=1)  # (B * n_patches, C, H, W)
    if mode == "train":
        model.train()

    contexts = {"train": contextlib.nullcontext, "eval": torch.no_grad}

    with contexts[mode]():
        logits_batch = model(inputs)

    if mode == "eval":
        logits_batch = logits_batch.unflatten(0, (bs, -1))  # (B, n_patches, C, H, W)
        logits_batch = patcher.merge_patches(logits_batch)  # (B, C, H, W)

    return logits_batch.squeeze()

def _eval_model_on_split(split, global_batch_size, model, data_dir):
    input_queue = build_input_queue(
          split=split,
          data_dir=data_dir,
          global_batch_size=global_batch_size,
          cycle=False,
          prefetch=True,
          )
    dice_metric = SurfaceDiceMetric(n_batches=len(input_queue), device=DEVICE)
    loss = 0
    n = 0
    
    for batch in tqdm(input_queue, desc="evaluting"):
        batch = {"inputs": batch[0], "targets": batch[1]}
        logits_batch = model_fn(model, batch, split, update_batch_norm=False)
        loss_batch = loss_fn(logits_batch, batch["targets"])
        predicted = torch.where(logits_batch >= PR_THRESHOLD, 1, 0)

        dice_metric.process_batch(predicted, batch["targets"])
        loss += loss_batch.sum().item()
        n += len(logits_batch)

    surface_dice = dice_metric.compute()
    loss /= n
    metrics = {
        "surface_dice": surface_dice,
        "loss": loss
        }
    return metrics
        
def eval_model(
            global_batch_size: int,
            model: nn.Module,
            data_dir: str,
            ) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    logging.info('Evaluating on the training split.')
    train_metrics = _eval_model_on_split(
        'eval_train',
        global_batch_size,
        model,
        data_dir,
        )
    eval_metrics = {'train/' + k: v for k, v in train_metrics.items()}

    # We always require a validation set.
    logging.info('Evaluating on the validation split.')
    validation_metrics = _eval_model_on_split(
        'validation',
        global_batch_size,
        model,
        data_dir)
    for k, v in validation_metrics.items():
      eval_metrics['validation/' + k] = v

    return eval_metrics