import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import numpy as np
import wandb

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    unique_token = f"{_config['name']}_seed{_config['seed']}_Griddly_{datetime.datetime.now()}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):
    if args.env == 'griddlygen':
        logging_env = 'Foraging'
    elif args.env == 'herding':
        logging_env = 'Herding'
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]}, # comment out for vmas
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        args.env,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    max_episode = 30000
    test_max_episode = 1000
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    last_test = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} episodes".format(max_episode))
    return_tracker = []
    avg_return_tracker = []
    regret_tracker = []
    avg_regret_tracker = []
    while episode <= max_episode:
        # print(runner.t_env)
        # Run for a whole episode at a time
        episode_batch, returns, regrets = runner.run(test_mode=False) # add regret for gathering
        return_tracker.extend(returns)
        regret_tracker.extend(regrets)
        if len(return_tracker) > 99:
            avg_return = np.mean(return_tracker)
            avg_regret = np.mean(regret_tracker)
            avg_regret_tracker.append(avg_regret)
            avg_return_tracker.append(avg_return)
            wandb.log({'Avg Training Return (' + args.name.upper() + ' ' + logging_env + ' ' + str(
                args.num_train_levels) + ' ' + 'train)': avg_return}, step=episode)
            if args.env == 'griddlygen':
                wandb.log({'Avg Training Regret (' + args.name.upper() + ' ' + logging_env + ' ' + str(
                    args.num_train_levels) + ' ' + 'train)': avg_regret}, step=episode)
            return_tracker = []
            regret_tracker = []
        buffer.insert_episode_batch(episode_batch)
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            # print('training...')
            learner.train(episode_sample, runner.t_env, episode)
        episode += args.batch_size_run

        if episode - last_test > 100:
            last_test = episode
            val_regret_tracker = []
            val_return_tracker = []
            # Change this range depending on batch size! 1 for mappo
            for i in range(11-args.batch_size_run):
                episode_batch, returns, regrets = runner.run(test_mode=True) # add regret for gathering
                val_regret_tracker.extend(regrets)
                val_return_tracker.extend(returns)
                if len(val_return_tracker) > 9:
                    avg_val_regret = np.mean(val_regret_tracker) # change to regret for gathering
                    avg_val_return = np.mean(val_return_tracker)
                    if args.env == 'griddlygen':
                        wandb.log({'Generalisation Gap (' + args.name.upper() + ' ' + logging_env + ' ' + str(
                            args.num_train_levels) + ' ' + 'train)': avg_regret_tracker[-1] - avg_val_regret}, step=episode)
                        wandb.log({'Avg Test Regret (' + args.name + ' ' + logging_env + ' ' + str(
                            args.num_train_levels) + ' ' + 'train)': avg_val_regret}, step=episode)
                    else:
                        wandb.log({'Generalisation Gap (' + args.name.upper() + ' ' + logging_env + ' ' + str(
                            args.num_train_levels) + ' ' + 'train)': avg_return_tracker[-1] - avg_val_return}, step=episode)
                    wandb.log({'Avg Test Return (' + args.name + ' ' + logging_env + str(
                        args.num_train_levels) + ' ' + 'train)': avg_val_return}, step=episode)
                    val_regret_tracker = []

        if args.save_model and (
                runner.t_env - model_save_time >= args.save_model_interval
                or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    logger.console_logger.info("Finished Training")

    return_tracker = []
    regret_tracker = []
    cur_episode = episode
    first_test = True
    if args.num_train_levels == 100:
        while episode <= (cur_episode + test_max_episode):

            episode_batch, returns, regrets = runner.run(test_mode=True, first_test=True) # add regrets for gathering
            first_test = False
            return_tracker.extend(returns)
            regret_tracker.extend(regrets)

            if len(return_tracker) > 99:
                avg_return = np.mean(return_tracker)
                avg_regret = np.mean(regret_tracker)
                wandb.log({'Avg Evaluation Return (' + args.name.upper() + ' ' + logging_env + ' ' + str(
                    args.num_train_levels) + ' ' + 'train)': avg_return}, step=episode-cur_episode)
                if args.env == 'griddlygen':
                    wandb.log({'Avg Evaluation Regret (' + args.name.upper() + ' ' + logging_env + ' ' + str(
                        args.num_train_levels) + ' ' + 'train)': avg_regret}, step=episode-cur_episode)
                return_tracker = []
                regret_tracker = []

            if args.save_model and (
                    runner.t_env - model_save_time >= args.save_model_interval
                    or model_save_time == 0
            ):
                model_save_time = runner.t_env
                save_path = os.path.join(
                    args.local_results_path, "models", args.unique_token, str(runner.t_env)
                )
                # "results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)

            episode += args.batch_size_run

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

        # runner.close_env()
        logger.console_logger.info("Finished Testing")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
                                          config["test_nepisode"] // config["batch_size_run"]
                                  ) * config["batch_size_run"]

    return config