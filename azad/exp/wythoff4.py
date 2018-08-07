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

        # # Treat (0,0) as a null move, and return what you were given
        # if available == (0, 0):
        #     i = locate_moves(available, self.all_possible_moves)
        #     return x, y, board, (0, 0), i, (0, 0), 1, True

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

        # # # Treat (0,0) as a null move, and return what you were given
        # if available == (0, 0):
        #     i = locate_moves(available, self.all_possible_moves)
        #     return x, y, board, (0, 0), i, (0, 0), 1, True

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
            self.episode += 1

            # Re-init
            done = False
            player_win = False
            opponent_win = False
            steps = 0
            reward = 0

            states = []
            xys = []
            rewards = []
            moves_i = []
            moves = []

            # Find starting position, and log it
            x, y, board, available = self.env.reset()
            board = flatten_board(board)

            states.append(board.unsqueeze(0))
            xys.append((x, y))

            if debug:
                print("---------------------------------------")
                print(">>> STUMBLER ({}).".format(self.episode))
                print(">>> Initial position ({}, {})".format(x, y))

            # ----------------------------------------------------------------
            while not done:
                # ------------------------------------------------------------
                # PLAYER
                (x, y, board, move, i, available, reward,
                 done) = self.player_step(x, y, board, available)
                if debug:
                    print(">>> Move player: {}.".format(move))

                steps += 1
                states.append(board.unsqueeze(0))
                xys.append((x, y))
                rewards.append(reward)
                moves.append(move)
                moves_i.append(i)

                if done:
                    # Note who wins to set the sign of the rewards during
                    # learning
                    player_win = True

                    # Create a null S x A -> S' transition
                    states.append(board.unsqueeze(0))
                    xys.append((x, y))
                    rewards.append(reward)
                    moves.append(move)
                    moves_i.append(i)

                    break
                # ------------------------------------------------------------
                # OPPONENT
                (x, y, board, move, i, available, reward,
                 done) = self.opponent_step(x, y, board, available)
                if debug:
                    print(">>> Move opponent: {}.".format(move))

                steps += 1
                states.append(board.unsqueeze(0))
                xys.append((x, y))
                rewards.append(reward)
                moves.append(move)
                moves_i.append(i)

                if done:
                    # Note who wins to set the sign of the rewards during
                    # learning
                    opponent_win = True

                    # Create a null S x A -> S' transition
                    states.append(board.unsqueeze(0))
                    xys.append((x, y))
                    rewards.append(reward)
                    moves.append(move)
                    moves_i.append(i)

                    break

            # ----------------------------------------------------------------
            # Unroll the state transitions, actions and rewards, so both agents
            # can learn from episode.
            states = th.cat(states)
            rewards = th.tensor(rewards).unsqueeze(1)
            moves_i = th.tensor(moves_i).unsqueeze(1)

            import ipdb
            ipdb.set_trace()

            # ---
            # PLAYER
            s_idx = th.range(0, steps - 1, 2).type(torch.LongTensor)
            S = states[s_idx]
            next_S = states[s_idx + 2]

            r_idx = th.range(0, steps - 1, 2).type(torch.LongTensor)
            R = rewards.gather(0, r_idx.unsqueeze(1))
            Mi = moves_i[r_idx]

            Qs = self.player(S).gather(1, Mi)
            max_Qs = self.player(next_S).max(1)[0].unsqueeze(1)

            # Reward sign depends on who wins...
            if opponent_win:
                R *= -1

            # Make prediction
            next_Qs = R.float() + (self.gamma * max_Qs)

            self.player_loss = F.l1_loss(Qs, next_Qs)
            self.player_optimizer.zero_grad()
            self.player_loss.backward()
            self.player_optimizer.step()

            # ---
            # OPPONENT
            s_idx = th.range(1, steps - 1, 2).type(torch.LongTensor)
            S = states[s_idx]
            next_S = states[s_idx + 2]

            r_idx = th.range(1, steps - 1, 2).type(torch.LongTensor)
            R = rewards.gather(0, r_idx.unsqueeze(1))
            Mi = moves_i[r_idx]

            Qs = self.opponent(S).gather(1, Mi)
            max_Qs = self.opponent(next_S).max(1)[0].unsqueeze(1)

            # Reward sign depends on who wins...
            if opponent_win:
                R *= -1

            # Make prediction
            next_Qs = R.float() + (self.gamma * max_Qs)

            self.opponent_loss = F.l1_loss(Qs, next_Qs)
            self.opponent_optimizer.zero_grad()
            self.opponent_loss.backward()
            self.opponent_optimizer.step()

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
                    os.path.join(tensorboard, 'steps'), steps, self.episode)
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
