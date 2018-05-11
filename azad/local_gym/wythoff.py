import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class Wythoff(gym.Env):
    def __init__(self, m, n, seed=None):
        # Board/pile size
        self.m = int(m)
        self.n = int(n)

        # Current postition
        # (needs a .reset()) to take
        # on useful values
        self.x = None
        self.y = None

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
        elif (self.x + dx == 0) and (self.y + dy == 0):
            self.x += dx
            self.y += dy

            reward = 1
            done = True
        # Move with no outcome
        else:
            self.x += dx
            self.y += dy
            reward = 0
            done = False

        # ------------------------------------
        # If the game ended the opponent 
        # can't play...
        if done:
            state = (self.x, self.y)
            return state, reward, done, info

        # ------------------------------------
        # The opponent plays...
        # perfectly.
        opt_dx, opt_dy = self.optimal_move()

        if ((self.x + opt_dx) == 0) and ((self.y + opt_dy) == 0):
            reward = -1
            done = True

            self.x += opt_dx
            self.y += opt_dy
        else:
            reward = 0
            done = False

            self.x += opt_dx
            self.y += opt_dy

        state = (self.x, self.y)

        return state, reward, done, info

    def optimal_move(self):
        # every step is followed by an optimal opponent play
        pass

    def reset(self):
        self.x = int(self.prng.randint(1, self.m + 1))
        self.y = int(self.prng.randint(1, self.n + 1))

    def render(self, mode='human', close=False):
        raise NotImplementedError("TODO")
        pass
