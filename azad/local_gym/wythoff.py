import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


def optimal_wythoff_move(state, m, n):
    # every step is followed by an optimal opponent play
    pass


def random_wythoff_move(state, m, n):
    pass


class WythoffEnv(gym.Env):
    """Wythoff's game.
    
    Note: the opponent is simulated by a perfect player
    """

    def __init__(self, m, n):
        # Board/pile size
        self.m = int(m)
        self.n = int(n)

        # Current postition
        # (needs a .reset()) to take
        # on useful values
        self.x = None
        self.y = None

        # Set by seed()
        self.prng = None

    def seed(self, seed):
        # Seed control
        self.prng = np.random.RandomState(seed)

    def step(self, move):
        # Empty. Required for gym API.
        info = {}

        # Parse the move
        dx, dy = move
        dx = int(dx)
        dy = int(dy)

        # ------------------------------------
        # Check for illegal moves,
        # return (-1, -1) for all

        # Can't move backward
        if (self.x + dx) <= self.x:
            reward = -1
            done = True
            state = (-1, -1)

        elif (self.y + dy) <= self.y:
            reward = -1
            done = True
            state = (-1, -1)

        # Out of bounds
        elif (self.x + dx) > self.m:
            reward = -1
            done = True
            state = (-1, -1)

        elif (self.y + dy) > self.n:
            reward = -1
            done = True
            state = (-1, -1)

        elif (self.x + dx) < 0:
            reward = -1
            done = True
            state = (-1, -1)

        elif (self.y + dy) < 0:
            reward = -1
            done = True
            state = (-1, -1)

        # Winning move?
        elif (self.x + dx == 0) and (self.y + dy == 0):
            self.x += dx
            self.y += dy
            state = (self.x, self.y)

            reward = 1
            done = True

        # Just move....
        else:
            self.x += dx
            self.y += dy
            state = (self.x, self.y)
            reward = 0
            done = False

        return state, reward, done, info

    def reset(self):
        if self.prng is None:
            self.seed(None)

        self.x = int(self.prng.randint(1, self.m))
        self.y = int(self.prng.randint(1, self.n))

        state = (self.x, self.y)
        return state

    def render(self, mode='human', close=False):
        raise NotImplementedError("TODO")


class Wythoff3x3(WythoffEnv):
    """A 3 by 3 Wythoff game"""

    def __init__(self):
        WythoffEnv.__init__(self, m=3, n=3)


class Wythoff10x10(WythoffEnv):
    """A 3 by 3 Wythoff game"""

    def __init__(self):
        WythoffEnv.__init__(self, m=10, n=10)