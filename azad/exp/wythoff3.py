"""Version 3: Game theory-informed RL models. """
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
from azad.local_gym.wythoff import locate_cold_moves
from azad.local_gym.wythoff import locate_best_move
from azad.local_gym.wythoff import locate_closest

from azad.models import Table
from azad.models import DeepTable3
from azad.models import HotCold2
from azad.models import HotCold3
from azad.models import ReplayMemory
from azad.policy import epsilon_greedy

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
from azad.util.wythoff import create_cold_board

from azad.util.wythoff import _np_epsilon_greedy
from azad.util.wythoff import _np_softmax
from azad.util.wythoff import _np_expected_value
from azad.util.wythoff import _np_greedy
from azad.util.wythoff import _np_plot_wythoff_max_values
from azad.util.wythoff import _np_plot_wythoff_min_values
from azad.util.wythoff import plot_cold_board
from azad.util.wythoff import plot_wythoff_board
from azad.util.wythoff import plot_wythoff_expected_values


class Jumpy(object):
    """A Wythoff-specific implementation of Policy Hill Climbing (PHC).

    Citation
    --------
    Bowling, M. & Veloso, M., 2001. Rational and convergent learning in 
    stochastic games. IJCAI International Joint Conference on Artificial 
    Intelligence, pp.1021–1026.
    """

    def __init__(self,
                 game,
                 delta=1.0,
                 gamma=0.98,
                 learning_rate=1e-3,
                 seed_value=None):
        # Init params
        self.delta = delta
        self.gamma = gamma
        self.learning_rate = learning_rate

        # Build the world
        self.game = game
        self.env = gym.envs.make(self.game)

        self.seed_value = seed_value
        self.env.seed(self.seed_value)

        self.m, self.n, _, _ = peek(env)
        self.all_possible_moves = create_all_possible_moves(m, n)

        # Build agents, and its optimizer
        default_Q = 0.0
        default_policy = 0.0
        self.player = {}
        self.oppenent = {}
        self.player_policy = {}
        self.oppenent_polcy = {}

        # Saved attrs
        self.done = None

    def _init_writer(self, tensorboard):
        if tensorboard is not None:
            try:
                os.makedirs(tensorboard)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

            self.writer = SummaryWriter(log_dir=tensorboard)

    def reset(self):
        """Reset the model and the env"""
        self.good = 0
        self.steps = 0

        x, y, board, moves = self.env.reset()

        return x, y, board, moves

    def good_move(self, move, x, y):
        """Was the move good?"""

        best_moves = locate_cold_moves(x, y)
        if move in best_moves:
            return True
        else:
            return False

    def player_step(self, board, moves):
        """Player makes a move, and learns from it"""
        # --------------------------------------------------------------------
        # Get Q(s, ...)
        try:
            Qs = self.player[board]
        except KeyError:
            self.player[board] = np.ones(len(moves)) * default_Q
            Qs = self.player[board]

        # Decide
        move_i = _np_softmax(Qs)
        move = moves[move_i]

        if self.good_move(move, x, y):
            self.good += 1

        # Freeze these
        grad_i = deepcopy(move_i)
        grad_board = deepcopy(board)
        Q = Qs[grad_i]

        # --------------------------------------------------------------------
        # Play, leading to s'
        (x, y, board, moves), reward, done, _ = env.step(move)
        board = tuple(flatten_board(board).numpy())

        # --------------------------------------------------------------------
        # Get max Q(s', a)
        try:
            max_Q = self.player[board].max()
            max_i = self.player[board].argmax()
        except KeyError:
            self.player[board] = np.ones(len(moves)) * default_Q
            max_Q = self.player[board].max()
            max_i = self.player[board].argmax()

        # Q update
        next_Q = reward + (self.gamma * max_Q)
        loss = next_Q - Q
        self.player[grad_board][grad_i] = Q + (self.learning_rate * loss)

        # Policy update
        if grad_i == max_i:
            self.player_policy[grad_board][grad_i] += self.delta
        else:
            self.player_policy[grad_board][
                grad_i] -= self.delta / (len(moves) - 1.0)

        self.steps += 1

        return board, moves, done

    def opponent_step(self, board, moves):
        """Opponent makes a move, and learns from it"""
        # --------------------------------------------------------------------
        # Get Q(s, ...)
        try:
            Qs = self.opponent[board]
        except KeyError:
            self.opponent[board] = np.ones(len(moves)) * default_Q
            Qs = self.opponent[board]

        # Decide
        move_i = _np_softmax(Qs)
        move = moves[move_i]

        if self.good_move(move, x, y):
            self.good += 1

        # Freeze these
        grad_i = deepcopy(move_i)
        grad_board = deepcopy(board)
        Q = Qs[grad_i]

        # --------------------------------------------------------------------
        # Play, leading to s'
        (x, y, board, moves), reward, done, _ = env.step(move)
        board = tuple(flatten_board(board).numpy())

        # --------------------------------------------------------------------
        # Get max Q(s', a)
        try:
            max_Q = self.opponent[board].max()
            max_i = self.opponent[board].argmax()
        except KeyError:
            self.opponent[board] = np.ones(len(moves)) * default_Q
            max_Q = self.opponent[board].max()
            max_i = self.opponent[board].argmax()

        # Q update
        next_Q = reward + (self.gamma * max_Q)
        loss = next_Q - Q
        self.opponent[grad_board][grad_i] = Q + (self.learning_rate * loss)

        # Policy update
        if grad_i == max_i:
            self.opponent_policy[grad_board][grad_i] += self.delta
        else:
            self.opponent_policy[grad_board][
                grad_i] -= self.delta / (len(moves) - 1.0)

        self.steps += 1

        return board, moves, done

    def train(self,
              num_episodes=10,
              render=False,
              tensorboard=None,
              update_every=5,
              seed_value=None,
              debug=False):
        """Learn the env."""
        # --------------------------------------------------------------------
        # Setup
        # Log?
        self._init_writer(tensorboard)

        # Show?
        if render:
            self.env = wrappers.Monitor(
                self.env, './tmp/{}'.format(self.env_name), force=True)

        # Control randomness
        self.env.seed(seed_value)

        # --------------------------------------------------------------------
        # !
        for trial in range(num_trials):
            x, y, board, moves = self.reset()

            done = False
            while not done:
                board, moves, done = self.player_step(board, moves)
                if not done:
                    board, moves, done = self.opponent_step(board, moves)

                # TODO tensor/debug


class JumpyPuppy(Jumpy):
    """A Wythoff-specific implementation of WoLF-PHC.

    Citation
    --------
    Bowling, M. & Veloso, M., 2001. Rational and convergent learning in 
    stochastic games. IJCAI International Joint Conference on Artificial 
    Intelligence, pp.1021–1026.
    """

    def __init__(self,
                 env,
                 delta_win=0.1,
                 delta_lose=1,
                 gamma=0.98,
                 learning_rate=1e-3):
        pass