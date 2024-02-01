import torch
from absl import logging
from glob import glob
from pytorch_utils import pytorch_setup
import os


_, _, DEVICE, _ = pytorch_setup()

def save_checkpoint(model, optimizer_state, train_state, global_step, eval_results, log_dir):
    checkpoint_path = os.path.join(log_dir, f"checkpoint_{global_step}")
    
    torch.save({
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_state["optimizer"].state_dict(),
            'scheduler_state_dict': optimizer_state["scheduler"].state_dict(),
            "train_state": train_state,
            "eval_results": eval_results
            }, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}.")
    
def maybe_load_checkpoint(model, optimizer_state, train_state, global_step, eval_results, log_dir):
    # latest checkpoint
    checkpoint_paths = glob(os.path.join(log_dir, 'checkpoint') + "*")
    if len(checkpoint_paths) == 0:
      return model, optimizer_state, train_state, global_step, eval_results 

    checkpoint_path = sorted(checkpoint_paths)[-1]
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(model, torch.nn.DataParallel):
      if checkpoint["model_state_dict"].keys()[0].starts_with("module"):
        model.load_state_dict(checkpoint['model_state_dict'])
      else: 
        model.load_state_dict({f"module.{k}": v for k, v in checkpoint['model_state_dict'].items()})
    else:
      if checkpoint["model_state_dict"].keys()[0].starts_with("module"):
        model.load_state_dict({k[len("module")]: v for k, v in checkpoint['model_state_dict'].items()})
      else:
        model.load_state_dict(checkpoint['model_state_dict'])



    optimizer_state["optimizer"].load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer_state["scheduler"].load_state_dict(checkpoint['scheduler_state_dict'])
    train_state = checkpoint['train_state']
    global_step = checkpoint['global_step']
    eval_results = checkpoint['eval_results']

    logging.info(f'Loaded checkpoint from {save_path}.') 

    return model, optimizer_state, train_state, global_step, eval_results