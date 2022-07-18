import numpy as np
import os
import random
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
from yaml import Loader
from run import run
import wandb

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# results_path = "/home/ubuntu/data"

@ex.main
def my_main(_run, _config, _log):
    for i in range(5):
        random_seed = np.random.randint(1111, 9999)
        level_seeds = np.random.randint(1111, 9999, 10000)
        print('Seed:', random_seed)
        logging_name = 'qmix-'+str(random_seed)
        wandb.init(project='marlgen', entity='jonnycook', name=logging_name, reinit=True)
        # Setting the random seed throughout the modules
        config = config_copy(_config)
        config["seed"] = random_seed
        config["level_seeds"] = level_seeds
        np.random.seed(config["seed"])
        th.manual_seed(config["seed"])
        # config['env_args']['seed'] = config["seed"]
        config['env_args']['state_last_action'] = False
        config['env_args']['seed'] = config["seed"]
        config['env_args']['level_seeds'] = config["level_seeds"]

        # run the framework
        config = {'runner': 'episode', 'mac': 'basic_mac', 'env': 'griddlygen',
                  'env_args': {'seed': random_seed, 'level_seeds': level_seeds}, 'batch_size_run': 1,
                  'test_nepisode': 100, 'test_interval': 50000, 'test_greedy': True, 'log_interval': 10000,
                  'runner_log_interval': 1000, 'learner_log_interval': 10000, 't_max': 20050000, 'use_cuda': True,
                  'buffer_cpu_only': True, 'use_tensorboard': False, 'save_model': False, 'save_model_interval': 50000,
                  'checkpoint_path': '', 'evaluate': False, 'load_step': 0, 'save_replay': False,
                  'local_results_path': 'results', 'gamma': 0.99, 'batch_size': 10, 'buffer_size': 5000, 'lr': 0.0005,
                  'optim_alpha': 0.99, 'optim_eps': 1e-05, 'grad_norm_clip': 10, 'add_value_last_step': True,
                  'agent': 'rnn', 'hidden_dim': 64, 'obs_agent_id': True, 'obs_last_action': False, 'repeat_id': 1,
                  'label': 'default_label', 'hypergroup': None, 'action_selector': 'epsilon_greedy',
                  'epsilon_start': 1.0, 'epsilon_finish': 0.05, 'epsilon_anneal_time': 5000, 'evaluation_epsilon': 0.0,
                  'mask_before_softmax': True, 'target_update_interval_or_tau': 200, 'obs_individual_obs': False,
                  'agent_output_type': 'q', 'learner': 'q_learner', 'entropy_coef': 0.01, 'standardise_returns': False,
                  'standardise_rewards': True, 'use_rnn': False, 'q_nstep': 5, 'critic_type': 'ac_critic', 'epochs': 4,
                  'eps_clip': 0.2, 'name': "qmix", 'seed': random_seed, 'mixing_embed_dim': 32, 'hypernet_layers': 2,
                  'hypernet_embed': 64, 'max_before_softmax': True, 'double_q': True, 'mixer': "qmix"}
        run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=Loader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    # params = deepcopy(sys.argv)
    params = ['src/main.py', '--config=qmix', '--env-config=griddlygen']
    th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=Loader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    # params = ["--config=vdn", "--env-config=gymma"]
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    # print(config_dict)
    # try:
    #     map_name = config_dict["env_args"]["map_name"]
    # except:
    #     map_name = config_dict["env_args"]["key"]


    # now add all the config to sacred
    ex.add_config(config_dict)

    # params = ["--config=vdn", "--env-config=gymma", "env_args.key=GDY-PredPreyEnv-v0"]
    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, f"sacred/{config_dict['name']}/Griddly")

    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    # ex.observers.append(MongoObserver())

    ex.run_commandline(params)

