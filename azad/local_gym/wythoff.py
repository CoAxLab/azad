import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class WythoffEnv(gym.Env):
    """Wythoff's game.
    
    Note: the opponent is simulated by a perfect player
    """

    def __init__(self, m, n):
        # Board/pile size
        self.m = int(m)
        self.n = int(n)

        self.info = {"m": m, "n": n}

        # Current postition
        # (needs a .reset()) to take
        # on useful values
        self.x = None
        self.y = None
        self.board = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(2)

    def step(self, move):
        # Empty. Required for gym API.

        # Parse the move
        dx, dy = move
        dx = int(dx)
        dy = int(dy)

        # ------------------------------------
        # Check for illegal moves,
        # return (-1, -1) for all

        # Can't move backward
        if (self.x + dx) > self.x:
            reward = -1

            # Don't move
            done = False
            state = (self.x, self.y)

        elif (self.y + dy) > self.y:
            reward = -1

            # Don't move
            done = False
            state = (self.x, self.y)

        # Out of bounds
        elif (self.x + dx) > self.m:
            reward = -1

            # Don't move
            done = False
            state = (self.x, self.y)

        elif (self.y + dy) > self.n:
            reward = -1

            # Don't move
            done = False
            state = (self.x, self.y)

        elif (self.x + dx) < 0:
            reward = -1

            # Don't move
            done = False
            state = (self.x, self.y)

        elif (self.y + dy) < 0:
            reward = -1

            # Don't move
            done = False
            state = (self.x, self.y)

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

        # Place the piece on the board
        self._reset_board()
        self.board[self.x, self.y] = 1
        state = (self.x, self.y, self.board)

        return state, reward, done, self.info

    def _reset_board(self):
        self.board = np.zeros((self.m, self.n))

    def reset(self):
        self.x = self.m - 1
        self.y = self.n - 1

        self._reset_board()
        self.board[self.x, self.y] = 1
        state = (self.x, self.y, self.board)

        return state

    def render(self, mode='human', close=False):
        pass


class Wythoff3x3(WythoffEnv):
    """A 3 by 3 Wythoff game"""

    def __init__(self):
        WythoffEnv.__init__(self, m=3, n=3)


class Wythoff10x10(WythoffEnv):
    """A 10 by 10 Wythoff game"""

    def __init__(self):
        WythoffEnv.__init__(self, m=10, n=10)