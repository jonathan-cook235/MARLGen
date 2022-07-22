import numpy as np
from griddly.util.rllib.environment.level_generator import LevelGenerator

class GeneralLevelGenerator(LevelGenerator):
    WALL = 'W'
    FORAGER = 'f'
    POTION = 'p'
    HOLE = 'h'

    def __init__(self, config, seed=None):
        super().__init__(config)
        self._width = config.get('width', 10)
        self._height = config.get('height', 10)
        self._max_potions = config.get('max_potions', 5)
        self._max_holes = config.get('max_holes', 5)
        self._num_agents = config.get('num_agents', 2)
        self._seed = seed

    def _place_walls(self, map):
        wall_y = np.array([0, self._height - 1])
        map[:, wall_y] = GeneralLevelGenerator.WALL

        wall_x = np.array([0, self._width - 1])
        map[wall_x, :] = GeneralLevelGenerator.WALL

        return map

    def _place_items(self, map, possible_locations, char, fixed, num):
        if not fixed:
            num_items = 1 + np.random.choice(num - 1)
        else:
            num_items = num
        for k in range(num_items):
            location_idx = np.random.choice(len(possible_locations))
            location = possible_locations[location_idx]
            del possible_locations[location_idx]
            if char == 'f':
                map[location[0], location[1]] = char + str(k+1)
            else:
                map[location[0], location[1]] = char

        return map, possible_locations, num_items

    def generate(self, level_seed):
        # np.random.seed(self._seed)
        np.random.seed(level_seed)
        map = np.chararray((self._width, self._height), itemsize=2)
        map[:] = '.'

        map = self._place_walls(map)

        possible_locations = []
        for w in range(1, self._width - 1):
            for h in range(1, self._height - 1):
                possible_locations.append([w, h])

        map, possible_locations, reward_max = self._place_items(
            map,
            possible_locations,
            GeneralLevelGenerator.POTION,
            False,
            self._max_potions
        )

        map, possible_locations, num_holes = self._place_items(
            map,
            possible_locations,
            GeneralLevelGenerator.HOLE,
            False,
            self._max_holes
        )

        map, possible_locations, num_agents = self._place_items(
            map,
            possible_locations,
            GeneralLevelGenerator.FORAGER,
            True,
            self._num_agents
        )

        level_string = ''
        for w in range(0, self._width):
            for h in range(0, self._height):
                level_string += map[w, h].decode().ljust(4)
            level_string += '\n'

        return level_string, reward_max

class HerdingLevelGenerator(LevelGenerator):
    WALL = 'W'
    DOG = 'd'
    SHEEP = 's'
    TARGET = 't'

    def __init__(self, config, seed=None):
        super().__init__(config)
        self._width = config.get('width', 10)
        self._height = config.get('height', 10)
        self._max_obstacles = config.get('max_obstacles', 5)
        self._num_sheep = config.get('num_sheep', 1)
        self._num_agents = config.get('num_agents', 3)
        self._num_targets = config.get('num_targets', 1)
        self._seed = seed

    def _place_walls(self, map):
        wall_y = np.array([0, self._height - 1])
        map[:, wall_y] = GeneralLevelGenerator.WALL

        wall_x = np.array([0, self._width - 1])
        map[wall_x, :] = GeneralLevelGenerator.WALL

        return map

    def _place_items(self, map, possible_locations, char, fixed, num):
        if not fixed:
            num_items = 1 + np.random.choice(num - 1)
        else:
            num_items = num
        for k in range(num_items):
            location_idx = np.random.choice(len(possible_locations))
            location = possible_locations[location_idx]
            del possible_locations[location_idx]
            if char == 'f':
                map[location[0], location[1]] = char + str(k+1)
            else:
                map[location[0], location[1]] = char

        return map, possible_locations

    def generate(self, level_seed):
        # np.random.seed(self._seed)
        np.random.seed(level_seed)
        map = np.chararray((self._width, self._height), itemsize=2)
        map[:] = '.'

        map = self._place_walls(map)

        possible_locations = []
        for w in range(1, self._width - 1):
            for h in range(1, self._height - 1):
                possible_locations.append([w, h])

        map, possible_locations = self._place_items(
            map,
            possible_locations,
            HerdingLevelGenerator.WALL,
            False,
            self._max_obstacles
        )

        map, possible_locations = self._place_items(
            map,
            possible_locations,
            HerdingLevelGenerator.TARGET,
            True,
            self._num_targets
        )

        map, possible_locations = self._place_items(
            map,
            possible_locations,
            HerdingLevelGenerator.SHEEP,
            True,
            self._num_sheep
        )

        map, possible_locations = self._place_items(
            map,
            possible_locations,
            GeneralLevelGenerator.DOG,
            True,
            self._num_agents
        )

        level_string = ''
        for w in range(0, self._width):
            for h in range(0, self._height):
                level_string += map[w, h].decode().ljust(4)
            level_string += '\n'

        return level_string