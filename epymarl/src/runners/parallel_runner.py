from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import wandb

# wandb.init(project='marlgen', entity='jonnycook')

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.testing = False
        self.first_test = False

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        count_train = 0
        count_test = 0
        # Change this to work for chosen batch size!
        # for griddly:
        train_interval = int(np.round(self.args.num_train_seeds/self.batch_size))
        test_interval = int(np.round(self.args.num_test_seeds / self.batch_size))
        for i in range(self.batch_size):
            lim_train = train_interval + (i*train_interval)
            lim_test = test_interval + (i*test_interval)
            # if self.args.env != 'vmas':
            env_args[i]["level_seeds"] = self.args.env_args["level_seeds"][count_train:lim_train]
            env_args[i]["test_seeds"] = self.args.env_args["test_seeds"][count_test:lim_test]
            count_train += train_interval
            count_test += test_interval

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.t = 0

        self.t_env = 0

        # remember to change this to whatever is in config as it is game-dependent!
        self.episode_limit = 200

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit+1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        # self.env.reset()
        # self.reset()
        self.parent_conns[0].send(("reset", [self.testing, self.first_test]))
        self.parent_conns[0].recv()
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()
        # Reset the envs
        # self.env.reset()
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", [self.testing, self.first_test]))

        pre_transition_data = {
            "state": [], # comment out for vmas
            "avail_actions": [],
            "obs": []
        }
        count = 0
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"]) # comment out for vmas
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
        self.batch.update(pre_transition_data, ts=0, env=self.args.env)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False, first_test=False):
        if test_mode:
            self.testing = True
        if first_test:
            self.first_test = True
        self.reset()
        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False, env=self.args.env)
            # print('ACTIONS CHOSEN:')
            # print(actions_chosen)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [], # comment out for vmas
                "avail_actions": [],
                "obs": []
            }
            # print('About to receive data back')
            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    # print('Receiving...')
                    data = parent_conn.recv()
                    # print('Got data')
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))
                    episode_returns[idx] += np.sum(data["reward"]) # see what changing to min at end of episode does
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"]) # comment out for vmas
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False,
                              env=self.args.env)
            # print('Batch updated')

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True,
                              env=self.args.env)

        # print('EPISODE RETURN:')
        # print(episode_return)
        # wandb.log({'episode return': episode_return})

        # comment out for vmas and herding
        if self.args.env == 'griddlygen':
            rewards_max = [0 for _ in range(self.batch_size)]
            for idx, parent_conn in enumerate(self.parent_conns):
                parent_conn.send(("get_max_reward", None))
                rewards_max[idx] = parent_conn.recv()
            regrets = [np.abs(episode_return - reward_max)
                       for episode_return, reward_max in zip(episode_returns, rewards_max)]
        else:
            regrets = []

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        # cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        # cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        # cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, episode_returns, regrets # comment out regrets for vmas and herding

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            # print('doing env step')
            obs, reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            # print('getting state, available actions and observations')
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            # print('sending back')
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state, # comment out for vmas
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            testing = data[0]
            first = data[1]
            env.reset(test_mode=testing, first_test=first)
            remote.send({
                "state": env.get_state(), # comment out for vmas
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            info = env.get_env_info()
            remote.send(info)
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "get_max_reward":
            reward_max = env.get_reward_max()
            remote.send(reward_max)
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

