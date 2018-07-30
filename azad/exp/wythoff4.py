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


class WythoffStumbler(object):
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
        self.avg_optim = 0.0
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
        self.num_actions = self.m * self.n

    def initialize_agents(self):
        # Player
        self.player = Table(self.m * self.n, self.num_actions)
        self.player_optimizer = optim.SGD(
            self.player.parameters(), lr=self.learning_rate)

        # Opponent
        self.opponent = Table(self.m * self.n, self.num_actions)
        self.opponent_optimizer = optim.SGD(
            self.opponent.parameters(), lr=self.learning_rate)

    def reset(self):
        """Reset the model and the env."""

        self.best = 0
        self.num_cold = 0

        x, y, board, available = self.env.reset()
        self.steps = 0

        return x, y, board, available

    def player_step(self, x, y, board, available):
        """Step the player's model"""

        # Set model
        self.model = self.player
        self.model_optimizer = self.player_optimizer

        # Step it
        x, y, board, move, i, available, reward, done = self._model_step(
            x, y, board, available)

        # Anlyze player's move
        if cold_move_available(x, y, available):
            self.num_cold += 1
            if move == locate_closest_cold_move(x, y, available):
                self.best += 1
        if self.num_cold > 0:
            self.avg_optim += (self.best - self.avg_optim) / (self.episode + 1)

        return x, y, board, move, i, available, reward, done

    def opponent_step(self, x, y, board, available):
        """Step the opponent's model"""

        # Set model
        self.model = self.opponent
        self.model_optimizer = self.opponent_optimizer

        # Step it
        return self._model_step(x, y, board, available)

    def epsilon_t(self):
        """Anneal epsilon with time (i.e., episode)"""

        # Tune exploration
        if self.anneal:
            epsilon_t = self.epsilon * (1.0 / np.log((self.episode + np.e)))
        else:
            epsilon_t = self.epsilon

        return epsilon_t

    def _model_step(self, x, y, board, available):
        """Step a model."""

        # Get all value
        Qs = self.model(board).detach()

        # Filter for available values
        moves_index = locate_moves(available, self.all_possible_moves)

        # Choose a move, wiht e-greedy
        with torch.no_grad():
            i = epsilon_greedy(Qs, self.epsilon_t(), index=moves_index)
            move = self.all_possible_moves[i]

        # Make the move
        (x, y, board, available), reward, done, _ = self.env.step(move)
        board = flatten_board(board)

        # -
        return x, y, board, move, i, available, reward, done

    def learn(self):
        """Learn"""
        # Q(s, a)
        Qs = self.model(self.state)
        self.Q = Qs.gather(0, torch.tensor(self.learner_move_i))

        # max Q(s', .)
        next_Qs = self.model(self.next_state).detach()
        self.max_Q = next_Qs.max()

        # Estimate value of next Q
        self.est_Q = (self.reward + (self.gamma * self.max_Q))

        # Learn
        self.loss = F.l1_loss(self.Q, self.est_Q)
        self.model_optimizer.zero_grad()
        self.loss.backward()
        self.model_optimizer.step()

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

            if debug:
                print("---------------------------------------")
                print(">>> STUMBLER ({}).".format(self.episode))
                print(">>> Initial position ({}, {})".format(x, y))

            # ----------------------------------------------------------------
            # Inital move by learner
            self.learner = self.player_step
            self.learner_available = available

            x, y, board, move, i, available, reward, done = self.learner(
                x, y, board, available)

            # Note returns
            self.reward = reward
            self.x, self.y = x, y
            self.next_state = board

            self.learner_board = board
            self.learner_move = move
            self.learner_move_i = i
            self.available = available

            if debug:
                print(">>> Move {}".format(move))

            # ----------------------------------------------------------------
            # Set the first 'mover'
            self.mover = self.opponent_step

            # ...then begin the mover/learner leap-frog loop
            while not done:
                # Mover moves
                x, y, board, move, i, available, reward, done = self.mover(
                    x, y, board, available)

                if debug:
                    print(">>> Move {}".format(move))

                # Note state vars
                self.reward = reward
                self.x, self.y = x, y
                self.next_state = board

                self.mover_board = board
                self.mover_move = move
                self.mover_move_i = i
                self.available = available

                # Mover wins are learner losses.
                if done:
                    self.reward *= -1

            # Learn
            self.learn()

            # Leap the frog: swap learner/mover
            self.learner, self.mover = self.mover, self.learner
            self.state = self.next_state
            self.learner_board = self.mover_board
            self.learner_move = self.mover_move
            self.learner_move_i = self.mover_move_i

            if debug:
                print(">>> Reward {}; max_Q {}; Loss(Q {}, est_Q {}) -> {}".
                      format(self.reward, self.max_Q,
                             float(self.Q.detach().numpy()), self.est_Q,
                             self.loss))

                grad_i = th.nonzero(self.model.fc1.weight.grad)
                print(">>> Player W grad: {}".format(
                    self.model.fc1.weight.grad[grad_i]))

                if done and (reward > 0):
                    print("*** WIN ***")
                if done and (reward < 0):
                    print("*** OPPONENT WIN ***")

            if tensorboard and (int(self.episode) % update_every) == 0:
                writer.add_graph(model, (board, ))

                writer.add_scalar(
                    os.path.join(tensorboard, 'reward'), self, reward,
                    self.episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'Q'), self.Q, self.episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'error'), self.loss,
                    self.episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'epsilon_t'), seld.epsilon_t(),
                    self.episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'steps'), self.steps,
                    self.episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'avg_optimal'), self.avg_optim,
                    self.episode)

                # Optimal ref:
                plot_cold_board(
                    self.m, self.n, path=tensorboard, name='cold_board.png')
                writer.add_image(
                    'True cold positions',
                    skimage.io.imread(
                        os.path.join(tensorboard, 'cold_board.png')))

                # EV:
                plot_wythoff_expected_values(
                    self.m,
                    self.n,
                    self.player,
                    path=tensorboard,
                    name='player_expected_values.png')
                writer.add_image(
                    'Max value',
                    skimage.io.imread(
                        os.path.join(tensorboard,
                                     'wythoff_expected_values.png')))

                est_hc_board = estimate_hot_cold(
                    self.m,
                    self.n,
                    self.player,
                    hot_threshold=0.5,
                    cold_threshold=0.0)
                plot_wythoff_board(
                    est_hc_board, path=tensorboard, name='est_hc_board.png')
                writer.add_image(
                    'Estimated hot and cold positions',
                    skimage.io.imread(
                        os.path.join(tensorboard, 'est_hc_board.png')))
