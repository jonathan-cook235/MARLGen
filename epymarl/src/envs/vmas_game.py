from vmas.simulator.environment import GymWrapper, Environment
from vmas import scenarios
import numpy as np
import os
import sys
import torch
import yaml
import gym

class Game4(GymWrapper):
    def __init__(self):
        self.scenario = scenarios.load('transport.py').Scenario()
        self.num_envs = 1
        self.device = 'cpu'
        self.continuous_actions = False
        self.episode_limit = 200
        self.env = Environment(
            self.scenario,
            self.num_envs,
            self.device,
            self.episode_limit,
            self.continuous_actions
        )
        super().__init__(self.env)

        self.n_actions = 4
        self.n_agents = 4
        self.active_agents = [0, 1, 2, 3]
        self.episode_count = 0
        self.episode_steps = 0
        self.total_steps = 0

        self.obs = None
        self.reward = 0

    def get_active_agents(self):
        return self.active_agents

    def step(self, actions):
        if torch.is_tensor(actions):
            actions = actions.numpy()
        assert len(actions) == self.n_agents

        self.total_steps += 1
        self.episode_steps += 1

        terminated = False

        self.obs, self.reward, terminated, info = super().step(actions)

        info = self.get_env_info()

        if self.episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self.episode_count += 1

        return self.obs, self.reward, terminated, info

    def get_obs(self):
        return self.obs

    def get_obs_size(self):
        obs = self.get_obs()
        size = obs[0].size()
        if len(size) > 1:
            obs_size = size[1] * len(obs)
        else:
            obs_size = size[0] * len(obs)
        return obs_size

    def get_state(self):
        return self.obs

    def get_state_size(self):
        state_size = self.get_obs_size()
        return state_size

    def get_avail_actions(self):
        avail_actions = [[1]*4 for i in range(self.n_agents)]
        return avail_actions

    def get_total_actions(self):
        return self.n_actions

    def reset(self, **kwargs):
        self.episode_steps = 0
        self.obs = super().reset()
        return self.obs

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info