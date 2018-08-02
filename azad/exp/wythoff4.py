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
from azad.local_gym.wythoff import locate_cold_moves

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
        self.steps = 0
        x, y, board, available = self.env.reset()

        return x, y, board, available

    def epsilon_t(self):
        """Anneal epsilon with time (i.e., episode)"""

        # Tune exploration
        if self.anneal:
            epsilon_t = self.epsilon * (1.0 / np.log((self.episode + np.e)))
        else:
            epsilon_t = self.epsilon

        return epsilon_t

    def _choose(self, Qs, available):
        """Step the env based on Q values"""

        # Filter for available values
        moves_index = locate_moves(available, self.all_possible_moves)

        # Choose a move, wiht e-greedy
        with torch.no_grad():
            i = epsilon_greedy(Qs, self.epsilon_t(), index=moves_index)
            move = self.all_possible_moves[i]

        return i, move

    def player_step(self, x, y, board, available):
        """Step the player's model"""

        # Choose a move
        Qs = self.player(board).detach()
        i, move = self._choose(Qs, available)

        # Anlyze the move
        if cold_move_available(x, y, available):
            best = 0
            if move in locate_cold_moves(x, y, available):
                best = 1
            self.avg_optim += (best - self.avg_optim) / (self.episode + 1)

        # Make the move
        (x, y, board, available), reward, done, _ = self.env.step(move)
        board = flatten_board(board)

        return x, y, board, move, i, available, reward, done

    def opponent_step(self, x, y, board, available):
        """Step the opponent's model"""
        # Choose a move
        Qs = self.opponent(board).detach()
        i, move = self._choose(Qs, available)

        # Make the move
        (x, y, board, available), reward, done, _ = self.env.step(move)
        board = flatten_board(board)

        return x, y, board, move, i, available, reward, done

    def learn(self, state, a, next_state, reward, model, optimizer):
        """Learn"""
        # Q(s, a)

        Qs = model(state)
        Q = Qs.gather(0, torch.tensor(a))

        # max Q(s', .)
        next_Qs = model(next_state).detach()
        max_Q = next_Qs.max()

        # Estimate value of next Q
        est_Q = (reward + (self.gamma * max_Q))

        # Learn
        loss = F.l1_loss(Q, est_Q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return Q, est_Q, loss, model, optimizer

    def train(self,
              num_episodes=3,
              render=False,
              tensorboard=None,
              update_every=5,
              debug=False,
              seed_value=None):
        """Learn the env."""

        self.reset()
        self._init_writer(tensorboard)

        # Reseed game?
        if seed_value is not None:
            self.seed_value = seed_value
            self.initialize_game(self.game, seed_value)

        # --------------------------------------------------------------------
        # !
        for episode in range(num_episodes):
            self.steps = 0
            self.episode += 1

            # We begin with a Restarting at the start.
            x, y, board, available = self.env.reset()
            board = flatten_board(board)

            if debug:
                print("---------------------------------------")
                print(">>> STUMBLER ({}).".format(self.episode))
                print(">>> Initial position ({}, {})".format(x, y))

            # At the begining we can not be done, and we must make at least
            # one move.
            null_action = False
            done = False
            while not done:
                # ------------------------------------------------------------
                # The player always moves first in a sequence

                # If there was a win last round, then the null_action let's
                # us learn from that.
                if null_action:
                    reward *= -1

                    # If we hit a null at the start, then done needs to be
                    # reset.
                    done = True
                    null_action = False

                    if debug:
                        print(">>> Move NULL")
                else:
                    # Shuffle, so we begin at s
                    state = board
                    state_xy = (x, y)

                    (x, y, board, move, i, available, reward,
                     done) = self.player_step(x, y, board, available)

                    # and end up at s'
                    next_state = board
                    next_state_xy = (x, y)
                    if debug:
                        print(">>> Move player: {}.".format(move))

                    # If player wins, set it up so that both opponent and player
                    # can learn from that.
                    if done:
                        reward *= -1
                        null_action = True

                        if debug:
                            print("*** Player won! ****")

                # ------------------------------------------------------------
                # Opponent learns now. From the past a_o (prior iteration)
                # and the present player move (right above).
                (Q, est_Q, loss,
                 self.opponent, self.opponent_optimizer) = self.learn(
                     state, i, next_state, reward, self.opponent,
                     self.opponent_optimizer)

                if done and debug:
                    print(">>> Opponent: s {} -> s'{}".format(
                        state_xy, next_state_xy))
                    print(
                        ">>> Opponent: reward {}, Loss(Q {}, est_Q {}) -> {}".
                        format(reward, float(Q.detach().numpy()), est_Q, loss))

                # If were done, and coming from a null state need to die...
                # to prevent an infinite learning loop.
                if done and not null_action:
                    break

                # ------------------------------------------------------------
                # Opponent plays now (unless were in the null_action state)
                if null_action:
                    reward *= -1

                    if debug:
                        print(">>> Move NULL.")
                else:
                    (x, y, board, move, _, available, reward,
                     done) = self.opponent_step(x, y, board, available)

                    # s'
                    next_state = board
                    next_state_xy = (x, y)

                    if debug:
                        print(">>> Move opponent: {}.".format(move))

                    # Opponent won, this is a player loss
                    if done:
                        reward *= -1
                        null_action = True

                        # We may need to loop round one more time...
                        # so temp set done to false
                        done = False

                        if debug:
                            print("*** Opponent won! ****")

                # ------------------------------------------------------------
                # Player learns now
                (Q, est_Q, loss,
                 self.player, self.player_optimizer) = self.learn(
                     state, i, next_state, reward, self.player,
                     self.player_optimizer)

                if (done or null_action) and debug:
                    print(">>> Player: s {} -> s'{}".format(
                        state_xy, next_state_xy))
                    print(">>> Player: reward {}; Loss(Q {}, est_Q {}) -> {}".
                          format(reward, float(Q.detach().numpy()), est_Q,
                                 loss))

                # Cheat for now... debug
                # self.opponent = self.player

                # Move count
                self.steps += 1

            # -------------------------------------------------------------
            if tensorboard and (int(self.episode) % update_every) == 0:
                writer = self.writer

                writer.add_graph(self.player, (state, ))
                writer.add_scalar(
                    os.path.join(tensorboard, 'reward'), reward, self.episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'Q'), Q, self.episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'error'), loss, self.episode)
                writer.add_scalar(
                    os.path.join(tensorboard, 'epsilon_t'), self.epsilon_t(),
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
                    'Player',
                    skimage.io.imread(
                        os.path.join(tensorboard,
                                     'player_expected_values.png')))

                plot_wythoff_expected_values(
                    self.m,
                    self.n,
                    self.opponent,
                    path=tensorboard,
                    name='opponent_expected_values.png')
                writer.add_image(
                    'Opponent',
                    skimage.io.imread(
                        os.path.join(tensorboard,
                                     'opponent_expected_values.png')))
