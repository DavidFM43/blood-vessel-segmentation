from absl import app
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from absl import flags
from absl import logging
from pytorch_utils import (
    pytorch_setup,
    _get_time,
    _get_time_ddp,
    _reset_cuda_mem,
    pytorch_init,
)
from profiler import Profiler, PassThroughProfiler
import datetime
import logger_utils
import torch.distributed as dist
from algorithm import init_optimizer_state, update_params
import halton
import struct
import json
import torch
import random_utils as prng
import workload


class TrainingCompleteError(Exception):
    pass


flags.DEFINE_string(
    "tuning_search_space",
    None,
    "The path to the JSON file describing the external tuning search space.",
)
flags.DEFINE_integer(
    "num_tuning_trials", 1, "The number of external hyperparameter trials to run."
)
flags.DEFINE_string("data_dir", None, "Dataset location.")
flags.DEFINE_string(
    "experiment_dir",
    None,
    "The root directory to store all experiments. "
    "It is required and the directory should have "
    "an absolute path rather than a relative path.",
)
flags.DEFINE_string("experiment_name", None, "Name of the experiment.")
flags.DEFINE_boolean(
    "save_intermediate_checkpoints",
    True,
    "Whether to save any intermediate checkpoints. "
    "If False, it will only keep the latest checkpoint.",
)
flags.DEFINE_boolean(
    "resume_last_run", None, "Whether to resume the experiment from its last run."
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
    "overwrite",
    False,
    "Whether to overwrite the experiment with identical experiment_dir and experiment_name.",
)
flags.DEFINE_boolean(
    "save_checkpoints", True, "Whether or not to checkpoint the model at every eval."
)
flags.DEFINE_integer(
    "rng_seed",
    None,
    "Value of rng seed. If None, a random seed will" "be generated from hardware.",
)
flags.DEFINE_boolean(
    "set_pytorch_max_split_size", False, "If true, set pytorch max_split_size_mb to 256"
)
FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()

if USE_PYTORCH_DDP:
    get_time = _get_time_ddp
else:
    get_time = _get_time

train_global_batch_size = 32

def main(_):
    profiler = Profiler() if FLAGS.profile else PassThroughProfiler()
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


def run_study(
    data_dir: str,
    profiler,
    max_global_steps: int,
    tuning_search_space: str,
    num_tuning_trials: int,
    log_dir: str,
    save_checkpoints: bool,
    rng_seed,
):
    if train_global_batch_size % N_GPUS != 0:
        raise ValueError(
            f"The global batch size ({train_global_batch_size}) has to be divisible by the number of GPUs ({N_GPUS})."
        )

    with open(tuning_search_space, "r", encoding="UTF-8") as search_space_file:
        tuning_search_space = halton.generate_search(
            json.load(search_space_file), num_tuning_trials
        )
    all_metrics = []
    for hi, hyperparameters in enumerate(tuning_search_space):
        if not rng_seed:
            rng_seed = struct.unpack("I", os.urandom(4))[0]
        logging.info(f"Using RNG seed {rng_seed}")
        rng = prng.PRNGKey(rng_seed)
        logging.info(f"--- Tuning run {hi + 1}/{num_tuning_trials} ---")

        tuning_dir_name = None
        if log_dir is not None:
            tuning_dir_name = os.path.join(log_dir, f"trial_{hi + 1}")
            logging.info(f"Creating tuning directory at {tuning_dir_name}.")
            logger_utils.makedir(tuning_dir_name)

            # If existing hyperparameter exists, use saved hyperparameters for consistency.
            hyperparameters = logger_utils.write_hparams(
                hyperparameters, tuning_dir_name
            )
            tuning_search_space[hi] = hyperparameters

        with profiler.profile("Train"):
            metrics = train_once(
                data_dir,
                rng,
                train_global_batch_size,
                profiler,
                hyperparameters,
                log_dir,
                rng_seed,
                max_global_steps,
                save_checkpoints,
            )
            all_metrics.append(metrics)
            logging.info(f'Tuning trial {hi + 1}/{num_tuning_trials}')
            logging.info(f'Hyperparameters: {tuning_search_space[hi]}')
            logging.info(f'Metrics: {all_metrics[hi]}')
            num_evals = len(all_metrics[hi]['eval_results'])
            logging.info(f'Total number of evals: {num_evals}')
            logging.info('=' * 20)
            score = min(metrics)
            return score

    return 0.99


def train_once(
    data_dir,
    rng,
    global_batch_size,
    profiler,
    hyperparameters,
    log_dir,
    rng_seed,
    max_global_steps,
    save_checkpoints,
):
    data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)
    logging.info("Initializing dataset.")
    with profiler.profile("Initializing dataset"):
        input_queue = workload.build_input_queue(
            rng=data_rng, split="train", data_dir=data_dir, global_batch_size=global_batch_size
        )
    logging.info("Initializing model.")
    with profiler.profile("Initializing model"):
        model = workload.init_model_fn(model_init_rng)
    with profiler.profile("Initializing optimizer"):
        optimizer_state = init_optimizer_state(max_global_steps, model, hyperparameters)
    logging.info("Initializing metrics bundle.")
    # Bookkeeping.
    train_state = {
        "is_time_remaining": True,
        "training_complete": False,
        "last_step_end_time": 0,
        "last_eval_time": 0,
        "accumulated_run_time": 0,
        "accumulated_eval_time": 0,
        "accumulated_logging_time": 0,
    }

    # Loggers and checkpoint setup.
    preemption_count = 0
    logging.info("Initializing checkpoint and logger.")
    if log_dir is not None:
        meta_file_name = os.path.join(log_dir, f"meta_data_{preemption_count}.json")
        logging.info(f"Saving meta data to {meta_file_name}.")
        meta_data = logger_utils.get_meta_data(workload, rng_seed)
        logger_utils.write_json(meta_file_name, meta_data)

        flag_file_name = os.path.join(log_dir, f"flags_{preemption_count}.json")
        logging.info(f"Saving flags to {flag_file_name}.")
        logger_utils.write_json(flag_file_name, flags.FLAGS.flag_values_dict())

        metrics_logger = logger_utils.set_up_loggers(
            log_dir, flags.FLAGS, hyperparameters
        )

    global_step = 0
    eval_results = []
    global_start_time = get_time()
    train_state["last_step_end_time"] = global_start_time

    logging.info("Starting training loop.")
    while train_state["is_time_remaining"] and not train_state["training_complete"]:
        step_rng = prng.fold_in(rng, global_step)
        data_select_rng, update_rng, eval_rng = prng.split(step_rng, 3)
        with profiler.profile("Data selection"):
            batch = next(input_queue)
        try:
            with profiler.profile("Update parameters"):
                optimizer_state, model = update_params(
                    model=model,
                    hyperparameters=hyperparameters,
                    batch=batch,
                    optimizer_state=optimizer_state,
                    global_step=global_step,
                    metrics_logger=metrics_logger,
                )
        except TrainingCompleteError:
            train_state["training_complete"] = True

        global_step += 1
        if (max_global_steps is not None) and (global_step == max_global_steps):
            train_state["training_complete"] = True

        # accumulate training time
        train_step_end_time = get_time()
        train_state["accumulated_run_time"] += (
            train_step_end_time - train_state["last_step_end_time"]
        )
        train_state["is_time_remaining"] = (
            train_state["accumulated_run_time"]
            < workload.max_allowed_runtime_sec
        )

        # evaluation
        if (
            train_step_end_time - train_state["last_eval_time"]
        ) >= workload.eval_period_time_sec or train_state["training_complete"]:
            with profiler.profile("Evaluation"):
                del batch
                _reset_cuda_mem()
                try:
                    eval_start_time = get_time()
                    latest_eval_result = workload.eval_model(model, data_dir)
                    # Save last eval time.
                    eval_end_time = get_time()
                    train_state["last_eval_time"] = eval_end_time

                    # Accumulate eval time.
                    train_state["accumulated_eval_time"] += (
                        eval_end_time - eval_start_time
                    )
                    eval_results.append((global_step, latest_eval_result))

                    logging_start_time = get_time()
                    if log_dir is not None:
                        metrics_logger.append_scalar_metrics(
                            latest_eval_result,
                            global_step=global_step,
                            preemption_count=preemption_count,
                            is_eval=True,
                        )

                    logging_end_time = get_time()
                    train_state["accumulated_logging_time"] += (
                        logging_end_time - logging_start_time
                    )

                    _reset_cuda_mem()
                except RuntimeError as e:
                    logging.exception(f"Eval step {global_step} error.\n")
                    if "out of memory" in str(e):
                        logging.warning(
                            "Error: GPU out of memory during eval during step "
                            f"{global_step}, error : {str(e)}."
                        )
                        _reset_cuda_mem()
        train_state['last_step_end_time'] = get_time()

    metrics = {'eval_results': eval_results, 'global_step': global_step}

    if log_dir is not None:
        metrics_logger.append_scalar_metrics(
            {'score': train_state['accumulated_submission_time']},
            global_step=global_step,
            preemption_count=preemption_count)
        metrics_logger.finish()

    return metrics


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("experiment_dir")
    flags.mark_flag_as_required("experiment_name")
    flags.mark_flag_as_required("tuning_search_space")
    flags.mark_flag_as_required("max_global_steps")
    app.run(main)
