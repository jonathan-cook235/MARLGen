from smac.env.multiagentenv import MultiAgentEnv
from griddly import GymWrapper as GriddlyGymWrapper
from griddly import GymWrapperFactory as Wrapper
from griddly import gd
import numpy as np
import os
import sys
import torch
import yaml
import gym

class Game1(GriddlyGymWrapper):
    def __init__(self, *args, seed=None):
        self.n_agents = 2
        self._seed = seed
        self.n_actions = 5
        self.grid_width = 9
        self.grid_height = 10
        # self.agent_view_size = 5
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.state = None
        self.observations = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.reward = 0
        self.episode_limit = 50

        self.action_map = {
            0: [0, 0],  # no-op
            1: [0, 1],  # left
            2: [0, 2],  # move
            3: [0, 3],  # right
            4: [1, 1],  # gather
        }

        yaml_filename = "gdy/simple_game.yaml"

        yaml_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), yaml_filename
        )

        with open(yaml_path, "r") as stream:
            yaml_dict = yaml.safe_load(stream)
            # Fixing the number of agents
            yaml_dict["Environment"]["Player"]["Count"] = self.n_agents
            # Fixing the agent view size
            yaml_dict["Environment"]["Player"]["Observer"][
                "Height"
            ] = self.grid_height
            yaml_dict["Environment"]["Player"]["Observer"][
                "Width"
            ] = self.grid_width
            # yaml_dict["Environment"]["Player"]["Observer"]["OffsetY"] = 0
            # yaml_dict["Environment"]["Player"]["Observer"]["OffsetY"] = (
            #     self.agent_view_size // 2
            # )
            self.yaml_string = yaml.dump(
                yaml_dict, default_flow_style=False, sort_keys=False
            )

        kwargs = {}
        kwargs["yaml_string"] = self.yaml_string
        kwargs["player_observer_type"] = kwargs.pop(
            "player_observer_type", gd.ObserverType.VECTOR
        )
        kwargs["global_observer_type"] = kwargs.pop(
            "global_observer_type", gd.ObserverType.BLOCK_2D
        )
        kwargs["max_steps"] = kwargs.pop("max_steps", 200)
        kwargs["environment_name"] = "Game1"

        super().__init__(*args, **kwargs)

    def step(self, actions):
        """Returns reward, terminated, info."""
        if torch.is_tensor(actions):
            actions = actions.numpy()
        assert len(actions) == self.n_agents

        self.last_action = np.eye(self.n_actions)[np.array(actions)]

        # agent_actions = []
        #
        # for a_id, action in enumerate(actions_int):
        #     agent_action = self.get_agent_action(a_id, action)
        #     if agent_action:
        #         agent_actions.append(agent_action)

        self._total_steps += 1
        self._episode_steps += 1

        terminated = False

        actions = [self.action_map[a] for a in actions]

        # print('MAPPED ACTIONS:')
        # print(actions)

        self.state, self.observations, self.reward, terminated, info = super().step(actions)
        # print('OBSERVATIONS:')
        # print(obs)
        # print('EPISODE STEP:')
        # print(self._episode_steps)

        if self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True

        if terminated:
            self._episode_count += 1

        return self.observations, self.reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        # agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return self.observations

    # def get_obs_agent(self, agent_id):
    #     """Returns observation for agent_id."""
    #     agent_obs = self.observations[agent_id]
    #     return agent_obs

    def get_obs_size(self):
        """Returns the size of the observation."""
        obs = self.get_obs()
        obs_array = np.array(obs)
        obs_size = np.size(obs_array)
        return obs_size

    def get_state(self):
        return self.state

    def get_state_size(self):
        """Returns the size of the global state."""
        state_size = len(self.get_state().flatten())
        return state_size

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        # avail_actions = [[1]*self.n_actions for _ in range(self.n_agents)]
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [0] + [1] * (self.n_actions - 1)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        state_obs = super().reset(global_observations=True)
        self.state = state_obs['global']
        self.observations = state_obs['player']
        # print(state.dtype)
        # return self.get_obs(), self.get_state()
        return self.observations, self.state

    def render(self):
        return super().render(observer="global", mode="rgb_array")

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        """Save a replay."""
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info
