from smac.env.multiagentenv import MultiAgentEnv
from griddly import GymWrapper as GriddlyGymWrapper
from griddly import GymWrapperFactory as Wrapper
from griddly.util.rllib.environment.level_generator import LevelGenerator
from griddly.RenderTools import VideoRecorder
from griddly import gd
import numpy as np
import os
import sys
import torch
import yaml
import gym
from .level_generator import HerdingLevelGenerator
import matplotlib.pyplot as plt

class Game3(GriddlyGymWrapper):
    def __init__(self, **kwargs):
        self.active_agents = [0, 1]
        self.active_sheep = 3
        self.n_agents = 2
        self._seed = kwargs['seed']
        self._level_seeds = kwargs['level_seeds']
        self._test_seeds = kwargs['test_seeds']
        self.variation = kwargs['variation']
        self.n_actions = 4
        self.agent_view_size = 8
        self.record_video = False
        self.recording_started = False
        self.video_filename = ""
        self._episode_count = 0
        self.validation_count = 0
        self._test_episode_count = 0
        self.test_count = 100000000
        self._episode_steps = 0
        self._total_steps = 0
        self.state = None
        self.observations = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.reward = 0
        self.episode_limit = 200

        self.action_map = {
            0: [0, 0],  # no-op
            1: [0, 1],  # left
            2: [0, 2],  # move
            3: [0, 3],  # right
            # 4: [1, 1],  # gather
        }

        yaml_filename = "gdy/herding_game.yaml"

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
            ] = self.agent_view_size
            yaml_dict["Environment"]["Player"]["Observer"][
                "Width"
            ] = self.agent_view_size
            self.yaml_string = yaml.dump(
                yaml_dict, default_flow_style=False, sort_keys=False
            )

        kwargs = {}
        kwargs["seed"] = self._seed
        kwargs["yaml_string"] = self.yaml_string
        kwargs["player_observer_type"] = kwargs.pop(
            "player_observer_type", gd.ObserverType.VECTOR
        )
        kwargs["global_observer_type"] = kwargs.pop(
            "global_observer_type", gd.ObserverType.VECTOR
        )
        kwargs["max_steps"] = kwargs.pop("max_steps", self.episode_limit)
        kwargs["environment_name"] = "Game3"
        kwargs["level"] = None
        # kwargs["level_seeds"] = self._level_seeds
        if self.variation:
            generator_config = {'min_width': 25, 'max_width': 40, 'min_height': 25, 'max_height': 40,
                                'max_obstacles': 40, 'num_agents': 2, 'num_sheep': 3, 'num_targets': 1}
        else:
            generator_config = {'min_width': 15, 'max_width': 15, 'min_height': 15, 'max_height': 15, 'max_obstacles': 5,
                            'num_agents': 2, 'num_sheep': 3, 'num_targets': 1}

        self.generator = HerdingLevelGenerator(generator_config, seed=self._seed)
        # kwargs["level"] = Generator.generate()
        # self.level = Generator.generate()

        super().__init__(**kwargs)

        # if self.record_video:
        if sys.platform != "linux":
            self.video_recorder = VideoRecorder()

    def get_active_sheep(self):
        state = super().get_state()
        active_sheep = set()
        count = 0
        for object in state["Objects"]:
            if object["Name"] == "sheep":
                count += 1
        # print('ACTIVE AGENTS')
        # print(count)
        return count

    def get_active_agents(self):
        return self.active_agents

    def step(self, actions, random=False):
        """Returns reward, terminated, info."""
        # if random:
        actions = np.random.randint(1, 4, 2)
        # print('in step')
        if torch.is_tensor(actions):
            actions = actions.numpy()
        assert len(actions) == self.n_agents

        self.last_action = np.eye(self.n_actions)[np.array(actions)]

        self._total_steps += 1
        self._episode_steps += 1
        # print(self._episode_steps)

        terminated = False

        # actions = [self.action_map[a] for a in actions]

        # print('doing super step')
        last_active_sheep = self.active_sheep
        self.observations, self.reward, terminated, info = super().step(actions)

        if self.record_video:
            frame = self.render(mode="rgb_array")
            if sys.platform == "linux":
                image = PIL.Image.fromarray(frame)
                image.save(
                    os.path.join(self.tmpdir, f"e_s_{self._total_steps}.png")
                )
            else:
                self.video_recorder.add_frame(frame)
        # print('checking sheep')
        # Terminate if no agents left
        self.active_sheep = self.get_active_sheep()
        if self.active_sheep < last_active_sheep:
            # print('Last Active Sheep:', last_active_sheep)
            # print('Active Sheep:', self.active_sheep)
            self.reward = [5, 5]

        if self.active_sheep < 1:
            terminated = True
            info["solved"] = True
            # self.reward = [5, 5]
        else:
            info["solved"] = False

        if self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True

        if terminated and self.record_video and self.recording_started:
            if sys.platform == "linux":
                gif_path = self.video_filename
                if gif_path == "":
                    gif_path = "herding.gif"
                elif gif_path.endswith("mp4"):
                    gif_path = gif_path[:-4] + ".gif"
                # Make the GIF and delete the temporary directory
                png_files = glob.glob(os.path.join(self.tmpdir, "e_s_*.png"))
                png_files.sort(key=os.path.getmtime)

                img, *imgs = [PIL.Image.open(f) for f in png_files]
                img.save(
                    fp=gif_path,
                    format="GIF",
                    append_images=imgs,
                    save_all=True,
                    duration=60,
                    loop=0,
                )
                shutil.rmtree(self.tmpdir)

                print(
                    "Saving replay GIF at {}".format(os.path.abspath(gif_path))
                )
            else:
                print("Closing video recorder.")
                self.video_recorder.close()
                self.recording_started = False

        if terminated:
            self._episode_count += 1

        return self.observations, self.reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        # agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        # print('OBSERVATIONS')
        # print(self.observations)
        return self.observations

    def get_obs_size(self):
        """Returns the size of the observation."""
        obs = self.get_obs()
        obs_array = np.array(obs)
        obs_size = np.size(obs_array)
        return obs_size

    def get_state(self):
        # print('STATE')
        # print(self.state[0][0])
        return super()._get_observation(self.game.observe(), self._global_observer_type)

    def get_state_size(self):
        """Returns the size of the global state."""
        state_size = len(self.get_state().flatten())
        return state_size

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        # print('ACTIVE AGENTS')
        # print(self.active_agents)
        # print('AGENT IDS')
        for agent_id in range(self.n_agents):
            # print(agent_id)
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        if agent_id not in self.active_agents:
            return [1] + [0] * (self.n_actions - 1)
        else:
            return [0] + [1] * (self.n_actions - 1)
            # return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self, record_video=False, test_mode=False, first_test=False, **kwargs):
        """Returns initial observations and states.
        :param **kwargs:
        """
        self.record_video = record_video
        if test_mode:
            level_seed = self._test_seeds[self.validation_count]
            if first_test:
                self.test_count = self.validation_count
            self.validation_count += 1
        else:
            level_seed = self._level_seeds[self._episode_count]
        test_count = self.validation_count - self.test_count
        if test_count > 0:
            if test_count < 200:
                self.episode_limit = 156
            elif 200 < test_count < 400:
                self.episode_limit = 225
            elif 400 < test_count < 600:
                self.episode_limit = 306
            elif 600 < test_count < 800:
                self.episode_limit = 400
        self.level, self.reward_max = self.generator.generate(level_seed, test_count, self.variation)
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        state_obs = super().reset(global_observations=True, level_string=self.level)
        self.state = state_obs['global']
        self.observations = state_obs['player']
        self.active_sheep = self.reward_max
        # print(state.dtype)
        # return self.get_obs(), self.get_state()
        # print(self._episode_count)
        if self.record_video and not self.recording_started:
            frame = self.render(mode="rgb_array")
            if sys.platform == "linux":
                # creating gifs not videos
                self.tmpdir = tempfile.mkdtemp()
                image = PIL.Image.fromarray(frame)
                image.save(
                    os.path.join(self.tmpdir, f"e_s_{self._total_steps}.png")
                )
            else:
                video_filename = self.video_filename
                if self.video_filename == "":
                    video_filename = "video_herding.mp4"
                print(f"Started recording at {video_filename}.")
                self.video_recorder.start(video_filename, frame.shape)
                self.video_recorder.add_frame(frame)
            self.recording_started = True
        return self.observations, self.state

    def render(self, **kwargs):
        return super().render(observer="global", mode="rgb_array")

    # def close(self):
    #     raise NotImplementedError

    def seed(self, **kwargs):
        os.environ['PYTHONHASHSEED'] = str(self._seed)
        np.random.seed(self._seed)
        return self._seed

    # def save_replay(self):
    #     """Save a replay."""
    #     raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info