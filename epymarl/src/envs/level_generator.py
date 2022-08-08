import numpy as np
from griddly.util.rllib.environment.level_generator import LevelGenerator

class GeneralLevelGenerator(LevelGenerator):
    WALL = 'W'
    FORAGER = 'f'
    POTION = 'p'
    HOLE = 'h'

    def __init__(self, config, seed=None):
        super().__init__(config)
        self._min_width = config.get('min_width', 30)
        self._max_width = config.get('max_width', 30)
        self._min_height = config.get('min_height', 30)
        self._max_height = config.get('max_height', 30)
        self._width = None
        self._height = None
        self._max_potions = config.get('max_potions', 10)
        self._max_holes = config.get('max_holes', 30)
        self._num_agents = config.get('num_agents', 4)
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
        if self._min_width != self._max_width:
            self._width = np.random.randint(self._min_width, self._max_width)
        else:
            self._width = self._min_width
        if self._min_height != self._max_height:
            self._height = np.random.randint(self._min_height, self._max_height)
        else:
            self._height = self._min_height
        map = np.chararray((self._max_width, self._max_height), itemsize=2)
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
            True,
            self._max_potions
        )

        map, possible_locations, num_holes = self._place_items(
            map,
            possible_locations,
            GeneralLevelGenerator.HOLE,
            True,
            self._max_holes
        )

        map, possible_locations, num_obstacles = self._place_items(
            map,
            possible_locations,
            GeneralLevelGenerator.WALL,
            True,
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
        for w in range(0, self._max_width):
            for h in range(0, self._max_height):
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
        self._min_width = config.get('min_width', 20)
        self._max_width = config.get('max_width', 20)
        self._min_height = config.get('min_height', 20)
        self._max_height = config.get('max_height', 20)
        self._width = None
        self._height = None
        self._max_obstacles = config.get('max_obstacles', 10)
        self._num_sheep = config.get('num_sheep', 1)
        self._num_agents = config.get('num_agents', 2)
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
            if char == 'd':
                map[location[0], location[1]] = char + str(k+1)
            elif char == 't':
                map[location[0], location[1]] = char
                if map[location[0]+1, location[1]] != 'W':
                    map[location[0]+1, location[1]] = char
                    x = 1
                else:
                    map[location[0]-1, location[1]] = char
                    x = -1
                if map[location[0], location[1]+1] != 'W':
                    map[location[0], location[1]+1] = char
                    y = 1
                else:
                    map[location[0], location[1]-1] = char
                    y = -1
                map[location[0]+x, location[1]+y] = char
            else:
                map[location[0], location[1]] = char

        return map, possible_locations

    def generate(self, level_seed):
        # np.random.seed(self._seed)
        np.random.seed(level_seed)
        if self._min_width != self._max_width:
            self._width = np.random.randint(self._min_width, self._max_width)
        else:
            self._width = self._min_width
        if self._min_height != self._max_height:
            self._height = np.random.randint(self._min_height, self._max_height)
        else:
            self._height = self._min_height
        map = np.chararray((self._max_width, self._max_height), itemsize=2)
        map[:] = '.'

        map = self._place_walls(map)

        possible_locations = []
        for w in range(1, self._width - 1):
            for h in range(1, self._height - 1):
                possible_locations.append([w, h])

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
            HerdingLevelGenerator.WALL,
            True,
            self._max_obstacles
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
            HerdingLevelGenerator.DOG,
            True,
            self._num_agents
        )

        level_string = ''
        for w in range(0, self._max_width):
            for h in range(0, self._max_height):
                level_string += map[w, h].decode().ljust(4)
            level_string += '\n'

        return level_string