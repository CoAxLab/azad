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


def epsilon_greedy(x, epsilon, index=None):
    """Pick the biggest, with probability epsilon"""

    # Filter x using index, but first ensure we can
    # map the action back to x' orignal 'space'
    if index is not None:
        x = x[index]

    if np.random.rand() < epsilon:
        action = np.random.randint(0, x.shape[0])
    else:
        action = np.argmax(x)

    # Map back to x's original space
    if index is not None:
        action = index[action]

    return action


class WythoffJumpy(object):
    """A Wythoff-specific implementation of Policy Hill Climbing (PHC).

    Citation
    --------
    Bowling, M. & Veloso, M., 2001. Rational and convergent learning in 
    stochastic games. IJCAI International Joint Conference on Artificial 
    Intelligence, pp.1021–1026.
    """

    def __init__(self,
                 game="Wythoff10x10",
                 delta=1.0,
                 gamma=0.98,
                 epsilon=0.1,
                 learning_rate=1e-3,
                 seed_value=None):

        # --------------------------------------------------------------------
        # Init params
        self.initialize_game(game, seed_value)

        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        # Build agents, and its optimizer
        self.default_Q = 0.0
        self.default_policy = 0.0
        self.player = {}
        self.oppenent = {}
        self.player_policy = {}
        self.oppenent_policy = {}

        # Setup up saved attrs
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

    def reset(self):
        """Reset the model and the env."""
        self.good = 0
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

    def good_move(self, move, x, y):
        """Was the move good?"""

        best_moves = locate_cold_moves(x, y)
        if move in best_moves:
            return True
        else:
            return False

    def player_step(self, x, y, board, moves):
        """Player makes a move, and learns from it"""

        self.mover = self.player
        self.mover_policy = self.player_policy
        x, y, board, moves, done = self._step(x, y, board, moves)

        return x, y, board, moves, done

    def opponent_step(self, x, y, board, moves):
        """Opponent makes a move, and learns from it"""

        self.mover = self.oppenent
        self.mover_policy = self.oppenent_policy
        x, y, board, moves, done = self._step(x, y, board, moves)

        return x, y, board, moves, done

    def _step(self, x, y, board, moves):
        """A private method for steppin'."""
        # --------------------------------------------------------------------
        # Get Q(s, ...)
        try:
            Qs = self.mover[board]
        except KeyError:
            self.mover[board] = np.ones(len(moves)) * self.default_Q
            Qs = self.mover[board]

        try:
            pis = self.mover_policy[board]
        except KeyError:
            self.mover_policy[board] = np.ones(
                len(moves)) * self.default_policy
            pis = self.mover_policy[board]

        # Decide
        move_i = epsilon_greedy(pis, self.epsilon)
        move = moves[move_i]

        if self.good_move(move, x, y):
            self.good += 1

        # Freeze these
        grad_i = deepcopy(move_i)
        grad_board = deepcopy(board)
        Q = Qs[grad_i]

        # --------------------------------------------------------------------
        # Play, leading to s'
        (x, y, board, moves), reward, done, _ = self.env.step(move)
        board = tuple(flatten_board(board).numpy())

        # --------------------------------------------------------------------
        # Get max Q(s', a)
        try:
            max_Q = self.mover[board].max()
            max_i = self.mover[board].argmax()
        except KeyError:
            self.mover[board] = np.ones(len(moves)) * self.default_Q
            max_Q = self.mover[board].max()
            max_i = self.mover[board].argmax()

        # Q update
        next_Q = reward + (self.gamma * max_Q)
        loss = next_Q - Q
        self.mover[grad_board][grad_i] = Q + (self.learning_rate * loss)

        # Policy update
        if grad_i == max_i:
            self.mover_policy[grad_board][grad_i] += self.delta
        else:
            self.mover_policy[grad_board][
                grad_i] -= self.delta / (len(self.all_possible_moves) - 1.0)

        # Set some attrs for later use.
        self.steps += 1

        self.move = move
        self.move_i = move_i
        self.grad_i = grad_i
        self.grad_board = grad_board
        self.board = board

        self.Q = Q
        self.max_Q = max_Q
        self.next_Q = next_Q
        self.reward = reward
        self.loss = loss

        return x, y, board, moves, done

    def train(self,
              num_episodes=3,
              render=False,
              tensorboard=None,
              update_every=5,
              debug=False,
              seed_value=None):
        """Learn the env."""

        self._init_writer(tensorboard)

        # Reseed game?
        if seed_value is not None:
            self.seed_value = seed_value
            self.initialize_game(self.game, seed_value)

        # --------------------------------------------------------------------
        # !
        for episode in range(num_episodes):
            # Restart
            x, y, board, moves = self.reset()
            board = tuple(flatten_board(board).numpy())

            if debug:
                print("---------------------------------------")
                print(">>> NEW GAME ({}).".format(episode))
                print(">>> Initial position ({}, {})".format(x, y))
                print(">>> Initial moves {}".format(moves))
                print("---------------------------------------")

            # Play
            done = False
            winner = 0
            while not done:
                # PLAYER
                x, y, board, moves, done = self.player_step(x, y, board, moves)
                if done:
                    winner = 1

                if debug:
                    print(">>> 1: PLAYER move {}".format(self.move))

                # OPPONENT
                if not done:
                    x, y, board, moves, done = self.opponent_step(
                        x, y, board, moves)
                if done:
                    winner = 2

                if debug:
                    print(">>> 2: OPPONENT move {}".format(self.move))

            # ----------------------------------------------------------------
            if debug:
                print(">>> Reward {}; Loss(Q {}, next_Q {}) -> {}".format(
                    self.reward, self.Q, self.next_Q, self.loss))

                if done and (self.reward > 0):
                    print("*** {} WINS ***".format(winner))

            # Log
            if tensorboard is not None and (int(episode) % update_every) == 0:
                writer = self.writer

                # Scalars
                writer.add_scalar(
                    os.path.join(tensorboard, 'winner'), winner, episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'fraction_good'),
                    self.good / float(self.steps), episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'winner_Q'), self.Q, episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'winner_loss'), self.loss,
                    episode)

                # Boards
                plot_cold_board(
                    self.m, self.n, path=tensorboard, name='cold_board.png')
                writer.add_image(
                    'cold_positions',
                    skimage.io.imread(
                        os.path.join(tensorboard, 'cold_board.png')))

                _np_plot_wythoff_max_values(
                    self.m,
                    self.n,
                    self.mover,
                    path=tensorboard,
                    name='wythoff_max_values.png')
                writer.add_image(
                    'winner_max_value',
                    skimage.io.imread(
                        os.path.join(tensorboard, 'wythoff_max_values.png')))


class WythoffJumpyPuppy(WythoffJumpy):
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