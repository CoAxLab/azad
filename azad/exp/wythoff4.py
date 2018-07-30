"""Version 4: Complete joint action updating. """
import os, csv
import sys

import errno
import pudb

from collections import defaultdict
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchviz import make_dot

import numpy as np
from scipy.constants import golden

import matplotlib.pyplot as plt
import seaborn as sns

import skimage
from skimage import data, io

import gym
from gym import wrappers
import azad.local_gym

from azad.local_gym.wythoff import create_moves
from azad.local_gym.wythoff import create_all_possible_moves
from azad.local_gym.wythoff import locate_moves
from azad.local_gym.wythoff import create_cold_board
from azad.local_gym.wythoff import create_board
from azad.local_gym.wythoff import cold_move_available
from azad.local_gym.wythoff import locate_closest_cold_move

from azad.models import Table

from azad.util.wythoff import peek
from azad.util.wythoff import pad_board
from azad.util.wythoff import flatten_board
from azad.util.wythoff import convert_ijv
from azad.util.wythoff import balance_ijv
from azad.util.wythoff import evaluate_models
from azad.util.wythoff import estimate_cold
from azad.util.wythoff import estimate_hot
from azad.util.wythoff import estimate_hot_cold
from azad.util.wythoff import estimate_alp_hot_cold
from azad.util.wythoff import estimate_strategic_value
from azad.util.wythoff import create_env
from azad.util.wythoff import create_moves
from azad.util.wythoff import create_board
from azad.util.wythoff import create_bias_board
from azad.util.wythoff import create_all_possible_moves
# from azad.util.wythoff import create_cold_board

from azad.util.wythoff import plot_cold_board
from azad.util.wythoff import plot_wythoff_board
from azad.util.wythoff import plot_wythoff_expected_values

import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical


def epsilon_greedy(x, epsilon, index=None):
    """Pick the biggest, with probability epsilon"""

    # Filter x using index, but first ensure we can
    # map the action back to x' orignal 'space'
    if index is not None:
        x = x[index]

    if th.rand(1) < epsilon:
        action = th.randint(0, x.shape[0], (1, ))
    else:
        action = th.argmax(x).unsqueeze(0)

    action = int(action)

    # Map back to x's original space
    if index is not None:
        action = index[action]

    return action


def softmax(x, beta=0.98, index=None):
    """Softmax policy"""
    # Filter x using index, but first ensure we can
    # map the action back to x' orignal 'space'
    if index is not None:
        x = x[index]

    probs = F.softmax(x * beta)
    m = Categorical(probs)
    action = m.sample()

    # Map back to x's original space
    if index is not None:
        action = index[action]

    return action


class WythoffJointActionStumbler(object):
    def __init__(self,
                 game="Wythoff10x10",
                 gamma=0.98,
                 epsilon=0.1,
                 learning_rate=1e-3,
                 anneal=False,
                 seed_value=None):

        # --------------------------------------------------------------------
        # Init params
        self.episode = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.anneal = anneal
        self.learning_rate = learning_rate

        self.initialize_game(game, seed_value)
        self.initialize_agents()
        self.reset()

    def _init_writer(self, tensorboard):
        if tensorboard is not None:
            try:
                os.makedirs(tensorboard)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

            self.writer = SummaryWriter(log_dir=tensorboard)

    def initialize_game(self, game, seed_value):
        self.game = game
        self.seed_value = seed_value
        self.env = create_env(game)
        self.env.seed(seed_value)

        self.m, self.n, _, _ = peek(self.env)
        self.all_possible_moves = create_all_possible_moves(self.m, self.n)
        self.num_actions = 2 * self.m * self.n

    def initialize_agents(self):
        self.num_actions = 2 * self.n * self.m
        self.player = Table(self.m * self.n, self.num_actions)
        self.opponent = Table(self.m * self.n, self.num_actions)

    def reset(self):
        """Reset the model and the env."""

        self.good = 0
        self.num_cold = 0
        self.avg_optim = 0.0
        self.steps = 0

        self.move = None
        self.move_i = None
        self.grad_i = None
        self.grad_board = None
        self.board = None

        self.Q = None
        self.max_Q = None
        self.next_Q = None
        self.reward = None
        self.loss = None

        x, y, board, moves = self.env.reset()

        return x, y, board, moves

    def player_step(self, x, y, board, available):
        """Step the player's model"""

        self.model = self.player
        return self._model_step(x, y, board, available)

    def opponent_step(self, x, y, board, available):
        """Step the opponent's model"""

        self.model = self.opponent
        return self._model_step(x, y, board, available)

    def _model_step(self, x, y, board, available):
        """Step a model."""

        # Tune exploration
        if self.anneal:
            epsilon_t = self.epsilon * (1.0 / np.log((self.episode + np.e)))
        else:
            epsilon_t = epsilon

        # Use value to choose a move
        moves_index = locate_moves(available, self.all_possible_moves)

        # Get value
        Qs = self.model(board).detach().view(self.m * self.n, self.m * self.n)
        import ipdb
        ipdb.set_trace()
        Qs = Qs.max(dim=1)

        with torch.no_grad():
            i = epsilon_greedy(Qs, epsilon_t, index=moves_index)
            move = self.all_possible_moves[i]

        # Make the move
        (x, y, board, available), reward, done, _ = env.step(move)
        board = flatten_board(board)

        # -
        return x, y, board, move, i, available, reward, done

    def learn(self):
        gamma = self.gamma
        state = self.state
        next_state = self.next_state
        u = self.u
        reward = self.reward
        available = self.learner_available
        moves_index = locate_moves(available, self.all_possible_moves)

        # Q(s, a)
        Qs = self.learner(state).detach().view(self.m * self.n,
                                               self.m * self.n)

        Qs = Qs.max(dim=1)  # Expectation over all mover's move
        Q = Qs.gather(0, torch.tensor(u))

        # max Q(s', .)
        next_Qs = self.model(next_state).detach().view(self.m * self.n,
                                                       self.m * self.n)
        next_Qs = next_Qs.max(dim=1)
        next_Qs.gather(0, th.tensor(moves_index))
        max_Q = next_Qs.max()

        # Estimate value difference
        est_Q = (reward + (gamma * max_Q))

        # Learn
        loss = F.l1_loss(Q, est_Q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self,
              num_episodes=3,
              render=False,
              tensorboard=None,
              update_every=5,
              debug=False,
              seed_value=None):
        """Learn the env."""

        pass

        self._init_writer(tensorboard)

        # Reseed game?
        if seed_value is not None:
            self.seed_value = seed_value
            self.initialize_game(self.game, seed_value)

        # --------------------------------------------------------------------
        # !
        for episode in range(num_episodes):
            self.episode += 1

            # Restart
            done = False
            x, y, board, available = self.reset()
            board = flatten_board(board)
            self.state = board

            # ----------------------------------------------------------------
            # Inital move by learner
            self.learner = self.player_step
            self.learner_available = available

            x, y, board, move, i, available, reward, done = self.learner(
                x, y, board, available)

            # Note returns
            self.next_state = board
            self.learner_board = board
            self.learner_move = move
            self.learner_move_i = i

            # If first move is a winner, generate a dummy 'u'
            # and learn from that
            if done:
                pass

            # Set the first 'mover'
            self.mover = self.opponent_step

            # ...then begin the mover/learner leap-frog loop
            while not done:
                # Mover moves
                x, y, board, move, i, available, reward, done = self.mover(
                    x, y, board, available)

                # Note state vars
                self.reward = reward
                self.next_state = board
                self.mover_board = board
                self.mover_move = move
                self.mover_move_i = i

                # Mover wins are learner losses.
                if done:
                    self.reward *= -1

                # Genetate u, the joint action index
                self.u = self.learner_move_i + self.mover_move_i

                # Learn
                self.learn()

                # Leap the frog: swap learner/mover
                self.learner, self.mover = self.mover, self.learner

                self.state = self.next_state
                self.learner_board = self.mover_board
                self.learner_move = self.mover_move
                self.learner_move_i = self.mover_move_i