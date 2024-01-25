from absl import app
import os
from absl import flags
from absl import logging
from pytorch_utils import pytorch_setup, _get_time, _get_time_ddp, _reset_cuda_mem, pytorch_init
from profiler import Profiler, PassThroughProfiler
import datetime
import logger_utils
import torch.distributed as dist
from algorithm import init_optimizer_state
import halton 
import json



flags.DEFINE_string(
    "tuning_search_space",
    None,
    "The path to the JSON file describing the external tuning search space."
)
flags.DEFINE_integer(
    "num_tuning_trials",
    1,
    "The number of external hyperparameter trials to run."
)
flags.DEFINE_string(
    "data_dir", 
    None, 
    "Dataset location."
)
flags.DEFINE_string(
    "experiment_dir",
    None,
    "The root directory to store all experiments. "
    "It is required and the directory should have "
    "an absolute path rather than a relative path.",
)
flags.DEFINE_string(
    "experiment_name",
    None,
    "Name of the experiment."
)
flags.DEFINE_boolean(
    "save_intermediate_checkpoints",
    True,
    "Whether to save any intermediate checkpoints. " "If False, it will only keep the latest checkpoint.",
)
flags.DEFINE_boolean(
    "resume_last_run", 
    None, 
    "Whether to resume the experiment from its last run."
)
flags.DEFINE_boolean(
    "append_timestamp",
    False,
    "If True, the current datetime will be appended to the experiment name. "
    "Useful for guaranteeing a unique experiment dir for new runs.",
)
flags.DEFINE_boolean("use_wandb", False, "Whether to use Weights & Biases logging.")
flags.DEFINE_boolean("profile", False, "Whether to produce profiling output.")
flags.DEFINE_integer("max_global_steps", None, "Maximum number of update steps.")
flags.DEFINE_boolean(
    "overwrite", False, "Whether to overwrite the experiment with identical experiment_dir and experiment_name."
)
flags.DEFINE_boolean("save_checkpoints", True, "Whether or not to checkpoint the model at every eval.")
flags.DEFINE_integer("rng_seed", None, "Value of rng seed. If None, a random seed will" "be generated from hardware.")
flags.DEFINE_boolean("set_pytorch_max_split_size", False, "If true, set pytorch max_split_size_mb to 256")
FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()

if USE_PYTORCH_DDP:
    get_time = _get_time_ddp
else:
    get_time = _get_time

def run_study(
        data_dir: str,
        profiler,
        max_global_steps: int,
        tuning_search_space: str,
        num_tuning_trials: int,
        log_dir: str,
        save_checkpoints: bool,
        rng_seed
):
    # Expand paths because '~' may not be recognized
    data_dir = os.path.expanduser(data_dir)
    train_global_batch_size = 10
    eval_global_batch_size = 10
    
    if train_global_batch_size % N_GPUS != 0:
        raise ValueError(f"The global batch size ({train_global_batch_size}) has to be divisible by the number of GPUs ({N_GPUS}).")
    if eval_global_batch_size % N_GPUS != 0:
        raise ValueError(f"The global eval batch size ({eval_global_batch_size}) has to be divisible by the number of GPUs ({N_GPUS}).")

    with open(tuning_search_space, "r", encoding="UTF-8") as search_space_file:
        tuning_search_space = halton.generate_search(json.load(search_space_file), num_tuning_trials)
    
    for hyperparameters in  enumerate(tuning_search_space):
        print(hyperparameters)

    return 0.99

def main(_):
    if FLAGS.profile:
        profiler = Profiler()
    else:
        profiler = PassThroughProfiler()

    pytorch_init(USE_PYTORCH_DDP, RANK, profiler)

    experiment_name = FLAGS.experiment_name
    if experiment_name and FLAGS.append_timestamp:
        experiment_name += datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    logging_dir_path = logger_utils.get_log_dir(
        FLAGS.experiment_dir, experiment_name, FLAGS.resume_last_run, FLAGS.overwrite
    )

    score = run_study(
        data_dir=FLAGS.data_dir,
        profiler=profiler,
        max_global_steps=FLAGS.max_global_steps,
        tuning_search_space=FLAGS.tuning_search_space,
        num_tuning_trials=FLAGS.num_tuning_trials,
        log_dir=logging_dir_path,
        save_checkpoints=FLAGS.save_checkpoints,
        rng_seed=FLAGS.rng_seed,
    )
    logging.info(f"Best validation score: {score}")

    if FLAGS.profile:
        logging.info(profiler.summary())

    if USE_PYTORCH_DDP:
        # Cleanup.
        dist.destroy_process_group()

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("experiment_dir")
    flags.mark_flag_as_required("experiment_name")
    flags.mark_flag_as_required("tuning_search_space")
    app.run(main)