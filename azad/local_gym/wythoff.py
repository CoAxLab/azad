import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


def optimal_move(state, m, n):
    # every step is followed by an optimal opponent play
    pass


def random_move(state, m, n):
    pass


class Wythoff(gym.Env):
    """Wythoff's game.
    
    Note: the opponent is simulated by a perfect player
    """

    def __init__(self, m, n, seed=None):
        # Board/pile size
        self.m = int(m)
        self.n = int(n)

        # Current postition
        # (needs a .reset()) to take
        # on useful values
        self.x = None
        self.y = None

        # Seed control
        self.prng = np.random.RandomState(seed)

    def step(self, move):
        # Empty. Required for gym API
        info = {}

        # Parse the move
        dx, dy = move
        dx = int(dx)
        dy = int(dy)

        # ------------------------------------
        # Check for illegal moves
        # Can't move backward
        if (self.x + dx) <= self.x:
            self.x += dx
            self.y += dy

            reward = -1
            done = True
        elif (self.y + dy) <= self.y:
            self.x += dx
            self.y += dy

            reward = -1
            done = True

        # Out of bounds
        elif (self.x + dx) > self.m:
            self.x += dx
            self.y += dy

            reward = -1
            done = True
        elif (self.y + dy) > self.n:
            self.x += dx
            self.y += dy

            reward = -1
            done = True

        elif (self.x + dx) < 0:
            self.x += dx
            self.y += dy

            reward = -1
            done = True
        elif (self.y + dy) < 0:
            self.x += dx
            self.y += dy

            reward = -1
            done = True

        # Winning move?
        elif (self.x + dx == 0) and (self.y + dy == 0):
            self.x += dx
            self.y += dy

            reward = 1
            done = True

        # Just move....
        else:
            self.x += dx
            self.y += dy
            reward = 0
            done = False

        # -
        state = (self.x, self.y)
        return state, reward, done, info

    def reset(self):
        self.x = int(self.prng.randint(1, self.m))
        self.y = int(self.prng.randint(1, self.n))

        state = (self.x, self.y)
        return state

    def render(self, mode='human', close=False):
        raise NotImplementedError("TODO")