from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
# import wandb

# wandb.init(project='marlgen', entity='jonnycook')

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.seed = self.env.seed()
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.testing = False
        self.first_test = False

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        self.env.reset(first_test=self.first_test)
        self.episode_limit = self.env.get_env_info()["episode_limit"]
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        # if self.t_env == 100:
        #     print('...........recording this episode...........')
        #     record_video = True
        # else:
        #     record_video = False
        self.env.reset(test_mode=self.testing, first_test=self.first_test)
        self.t = 0

    def run(self, test_mode=False, first_test=False):
        if test_mode:
            self.testing = True
        if first_test:
            self.first_test = True
        self.reset()
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        returns = []
        regrets = []

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()], # comment out for vmas
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t, env=self.args.env)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()
            # print('ACTIONS:')
            # print(actions)
            obs, reward, terminated, env_info = self.env.step(cpu_actions[0])
            # for griddly games:
            if self.args.env != 'vmas':
                self.env.render(observer='global')
            # print('OBSERVATIONS')
            # print(obs)
            episode_return += np.sum(reward)
            # if self.t_env == 100:
            #     print(episode_return)

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t, env=self.args.env)

            self.t += 1

        returns.append(episode_return)
        # for griddly gathering
        if self.args.env == 'griddlygen':
            reward_max = self.env.get_reward_max()
            regrets.append(np.abs(episode_return - reward_max))

        # print('EPISODE RETURN:')
        # print(episode_return)
        # wandb.log({'episode return': episode_return})

        last_data = {
            "state": [self.env.get_state()], # comment out for vmas
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        # print('LAST OBS:')
        # print(last_data["obs"])
        self.batch.update(last_data, ts=self.t, env=self.args.env)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t, env=self.args.env)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        # cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        # cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            # wandb.log({'avg episode return': np.mean(returns)})
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, returns, regrets # add regrets for gathering

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
