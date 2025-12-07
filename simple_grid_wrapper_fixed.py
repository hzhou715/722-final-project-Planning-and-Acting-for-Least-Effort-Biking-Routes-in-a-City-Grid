# ✅ simple_grid_wrapper_fixed.py (corrected attribute names)
import gymnasium as gym
import numpy as np
import gym_simplegrid

class SimpleGridWrapper:
    def __init__(self, seed=42, start=(0, 0), goal=(7, 7), blocked_cells=None, terrain=None):
        self.seed = seed
        self.size = 8
        self.env = gym.make("SimpleGrid-v0", obstacle_map="8x8")

        # Do a single reset with the correct options
        self.obs, _ = self.env.reset(seed=seed, options={"start_loc": start, "goal_loc": goal})
        
        # Get obstacles and validate positions (use 'obstacles' not 'obstacle_map')
        obstacles = self.env.unwrapped.obstacles
        if obstacles[start]:
            raise ValueError(f"Start position {start} overlaps with wall.")
        if obstacles[goal]:
            raise ValueError(f"Goal position {goal} overlaps with wall.")
        
        self.agent_pos = tuple(self.env.unwrapped.agent_xy)
        self.goal_pos = tuple(self.env.unwrapped.goal_xy)

        self.terrain = terrain if terrain is not None else self._generate_terrain()
        self.walls = self._get_walls()

        self.blocked = np.zeros((self.size, self.size), dtype=bool)
        if blocked_cells:
            for x, y in blocked_cells:
                self.blocked[x, y] = True

    def _generate_terrain(self):
        rng = np.random.default_rng(self.seed)
        return rng.integers(1, 6, size=(self.size, self.size))

    def _get_walls(self):
        walls = np.zeros((self.size, self.size), dtype=bool)
        if hasattr(self.env.unwrapped, 'obstacles'):
            obstacles = self.env.unwrapped.obstacles
            for i in range(self.size):
                for j in range(self.size):
                    if obstacles is not None and obstacles[i, j]:
                        walls[i, j] = True
        return walls

    def get_state(self):
        return {
            "agent": self.agent_pos,
            "goal": self.goal_pos,
            "terrain": self.terrain,
            "walls": self.walls,
            "blocked": self.blocked,
            "size": self.size
        }

    def render(self):
        self.env.render()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.agent_pos = tuple(self.env.unwrapped.agent_xy)
        return obs, reward, terminated, truncated, info