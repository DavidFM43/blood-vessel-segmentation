from input_pipeline import KidneyDataset
import pytorch_utils
import data_utils
import albumentations as A
import torch


USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()
patch_size = 224


def build_input_queue(rng, split, data_dir, global_batch_size):
    is_train = split == 'train'
    train_transforms = A.Compose([A.RandomCrop(patch_size, patch_size)])
    eval_transforms = None
    transforms = train_transforms if is_train else eval_transforms

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
            ds, num_replicas=N_GPUS, rank=RANK, shuffle=True)
      else:
        sampler = data_utils.DistributedEvalSampler(
            ds, num_replicas=N_GPUS, rank=RANK, shuffle=False)
            
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=ds_iter_batch_size,
        shuffle=not USE_PYTORCH_DDP and is_train,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=is_train
        )
    dataloader = data_utils.PrefetchedWrapper(dataloader, DEVICE)
    dataloader = data_utils.cycle(dataloader, custom_sampler=USE_PYTORCH_DDP, use_mixup=False)

    return dataloader

