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
from azad.local_gym.wythoff import locate_all_cold_moves
from azad.local_gym.wythoff import locate_closest_cold_move
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


def observe_actions(path,
                    num_trials,
                    epsilon=0.1,
                    gamma=0.8,
                    learning_rate=0.1,
                    game='Wythoff10x10',
                    tensorboard=False,
                    update_every=100,
                    debug=False,
                    seed=None):
    # ------------------------------------------------------------------------
    # Setup

    # Create path
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    if tensorboard:
        writer = SummaryWriter(log_dir=path)

    # Create env
    env = create_env(game)
    env.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------------
    # Build a Q agent, and its optimizer
    m, n, board, _ = peek(env)
    all_possible_moves = create_all_possible_moves(m, n)

    # Init the lookup table
    default_Q = 0.0
    player = {}
    opponent = {}

    # Run some trials
    for trial in range(num_trials):
        # ----------------------------------------------------------------
        # Game re-init
        steps = 0
        good_plays = 0
        x, y, board, moves = env.reset()

        # State re-init
        last_move_board = board
        state = np.outer(board.flatten(), last_move_board.flatten())
        state = tuple(state.flatten())

        if debug:
            print("---------------------------------------")
            print(">>> NEW GAME ({}).".format(trial))
            print(">>> Initial position ({}, {})".format(x, y))
            print(">>> Initial moves {}".format(moves))
            print("---------------------------------------")

        done = False
        bad_move = False
        while not done:
            # ----------------------------------------------------------------
            # PLAYER

            # Get Q(s, ...)
            try:
                Qs = player[state]
            except KeyError:
                player[state] = np.ones(len(moves)) * default_Q
                Qs = player[state]

            # Decide
            # move_i = _np_epsilon_greedy(Qs, epsilon=epsilon)
            move_i = _np_softmax(Qs)
            move = moves[move_i]

            # Find best move, as a reference
            best = locate_closest_cold_move(x, y, moves)
            if best is not None:
                steps += 1

            bonus = 0
            if move == best:
                good_plays += 1
                bonus = 1
            else:
                bonus = -1

            # Freeze these
            grad_i = deepcopy(move_i)
            grad_state = deepcopy(state)
            Q = Qs[grad_i]

            # Play, leading to s'
            if debug:
                print(">>> PLAYER move {} (best {})".format(move, best))

            (x, y, board, moves), reward, done, _ = env.step(move)
            state = np.outer(board.flatten(), last_move_board.flatten())
            state = tuple(state.flatten())

            # Get max Q(s', a)
            try:
                max_Q = player[state].max()
            except KeyError:
                player[state] = np.ones(len(moves)) * default_Q
                max_Q = player[state].max()

            reward += bonus

            next_Q = reward + (gamma * max_Q)
            loss = next_Q - Q
            player[grad_state][grad_i] = Q + (learning_rate * loss)

            # ----------------------------------------------------------------
            last_move_board = create_board(move[0], move[1], m, n)

            # ----------------------------------------------------------------
            if debug:
                print(">>> Reward {}; Loss(Q {}, next_Q {}) -> {}".format(
                    reward, Q, next_Q, loss))

                if done and (reward > 0):
                    print("*** WIN ***")

            if tensorboard and (int(trial) % update_every) == 0:
                writer.add_scalar(os.path.join(path, 'player_Q'), Q, trial)
                writer.add_scalar(
                    os.path.join(path, 'player_error'), loss.data[0], trial)
                writer.add_scalar(
                    os.path.join(path, 'player_steps'), steps, trial)

                frac = 0.0
                if steps > 0:
                    frac = good_plays / float(steps)
                writer.add_scalar(
                    os.path.join(path, 'player_good_plays'), frac, trial)

            # ----------------------------------------------------------------
            # End of game? Player won.
            if not done:
                # ------------------------------------------------------------
                # OPPONENT
                state = np.outer(board.flatten(), last_move_board.flatten())
                state = tuple(state.flatten())

                # Get Q(s, ...)
                try:
                    Qs = opponent[state]
                except KeyError:
                    opponent[state] = np.ones(len(moves)) * default_Q
                    Qs = opponent[state]

                # Decide
                move_i = _np_epsilon_greedy(Qs, epsilon=0.0)
                move = moves[move_i]

                # Freeze these
                grad_i = deepcopy(move_i)
                grad_state = deepcopy(state)
                Q = Qs[grad_i]

                if debug:
                    print(">>> OPPONENT move {} (best {})".format(move, best))

                # Play, leading to s'
                (x, y, board, moves), reward, done, _ = env.step(move)
                state = np.outer(board.flatten(), last_move_board.flatten())
                state = tuple(state.flatten())

                # Get max Q(s', a)
                try:
                    max_Q = opponent[state].max()
                except KeyError:
                    opponent[state] = np.ones(len(moves)) * default_Q
                    max_Q = opponent[state].max()

                next_Q = reward + (gamma * max_Q)
                loss = next_Q - Q
                opponent[grad_state][grad_i] = Q + (learning_rate * loss)

                # ------------------------------------------------------------
                last_move_board = create_board(move[0], move[1], m, n)

    # ------------------------------------------------------------------------
    # The end
    if tensorboard:
        writer.close()

    return player, opponent, env


def independent(path,
                num_trials,
                epsilon=0.1,
                gamma=0.8,
                learning_rate=0.1,
                game='Wythoff10x10',
                tensorboard=False,
                update_every=100,
                debug=False,
                seed=None):
    # ------------------------------------------------------------------------
    # Setup

    # Create path
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    if tensorboard:
        writer = SummaryWriter(log_dir=path)

    # Create env
    env = create_env(game)
    env.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------------
    # Build a Q agent, and its optimizer
    m, n, board, _ = peek(env)
    all_possible_moves = create_all_possible_moves(m, n)

    # Init the lookup table
    default_Q = 0.0
    player = {}
    opponent = {}

    # Run some trials
    for trial in range(num_trials):
        # ----------------------------------------------------------------
        # Re-init
        steps = 0
        good_plays = 0

        x, y, board, moves = env.reset()
        board = tuple(flatten_board(board).numpy())

        if debug:
            print("---------------------------------------")
            print(">>> NEW GAME ({}).".format(trial))
            print(">>> Initial position ({}, {})".format(x, y))
            print(">>> Initial moves {}".format(moves))
            print("---------------------------------------")

        done = False
        while not done:
            # ----------------------------------------------------------------
            # PLAYER

            # Get Q(s, ...)
            try:
                Qs = player[board]
            except KeyError:
                player[board] = np.ones(len(moves)) * default_Q
                Qs = player[board]

            # Decide
            move_i = _np_softmax(Qs)
            # move_i = np.random.randint(0, len(moves) + 1)
            move = moves[move_i]

            # Find best move
            best = locate_closest_cold_move(x, y, moves)
            if best is not None:
                steps += 1
            if move == best:
                good_plays += 1

            if best is None:
                best = locate_closest(moves)

            # Freeze these
            grad_i = deepcopy(move_i)
            grad_board = deepcopy(board)
            Q = Qs[grad_i]

            # Play, leading to s'
            if debug:
                print(">>> PLAYER move {} (best {})".format(move, best))

            (x, y, board, moves), reward, done, _ = env.step(move)
            board = tuple(flatten_board(board).numpy())

            # Get max Q(s', a)
            try:
                max_Q = player[board].max()
            except KeyError:
                player[board] = np.ones(len(moves)) * default_Q
                max_Q = player[board].max()

            next_Q = reward + (gamma * max_Q)
            loss = next_Q - Q
            player[grad_board][grad_i] = Q + (learning_rate * loss)

            # End of game? Player won.
            if done:
                break

            # ----------------------------------------------------------------
            # OPPONENT

            # Get Q(s, ...)
            try:
                Qs = opponent[board]
            except KeyError:
                opponent[board] = np.ones(len(moves)) * default_Q
                Qs = opponent[board]

            # Decide
            move_i = _np_epsilon_greedy(Qs, epsilon=0.0)
            # move_i = _np_softmax(Qs)
            move = moves[move_i]

            # Find best move
            best = locate_closest_cold_move(x, y, moves)
            if best is not None:
                steps += 1
            if move == best:
                good_plays += 1

            if best is None:
                best = locate_closest(moves)

            # Freeze these
            grad_i = deepcopy(move_i)
            grad_board = deepcopy(board)
            Q = Qs[grad_i]

            if debug:
                print(">>> OPPONENT move {} (best {})".format(move, best))

            # Play, leading to s'
            (x, y, board, moves), reward, done, _ = env.step(move)
            board = tuple(flatten_board(board).numpy())

            # Get max Q(s', a)
            try:
                max_Q = opponent[board].max()
            except KeyError:
                opponent[board] = np.ones(len(moves)) * default_Q
                max_Q = opponent[board].max()

            next_Q = reward + (gamma * max_Q)
            loss = next_Q - Q
            opponent[grad_board][grad_i] = Q + (learning_rate * loss)

            # Keep count of game moves.
            # steps += 1

            # ----------------------------------------------------------------
            if debug:
                print(">>> Reward {}; Loss(Q {}, next_Q {}) -> {}".format(
                    reward, Q, next_Q, loss))

                if done and (reward > 0):
                    print("*** WIN ***")
                if done and (reward < 0):
                    print("*** OPPONENT WIN ***")

            if tensorboard and (int(trial) % update_every) == 0:
                writer.add_scalar(os.path.join(path, 'reward'), reward, trial)
                writer.add_scalar(os.path.join(path, 'Q'), Q, trial)
                writer.add_scalar(
                    os.path.join(path, 'error'), loss.data[0], trial)
                writer.add_scalar(os.path.join(path, 'steps'), steps, trial)

                frac = 0.0
                if steps > 0:
                    frac = good_plays / float(steps)
                writer.add_scalar(
                    os.path.join(path, 'good_plays'), frac, trial)

                _np_plot_wythoff_max_values(
                    m, n, player, path=path, name='player_max_values.png')
                writer.add_image(
                    'player_max_value',
                    skimage.io.imread(
                        os.path.join(path, 'player_max_values.png')))

                _np_plot_wythoff_max_values(
                    m, n, opponent, path=path, name='opp_max_values.png')
                writer.add_image(
                    'opponent_max_value',
                    skimage.io.imread(
                        os.path.join(path, 'opp_max_values.png')))

                _np_plot_wythoff_min_values(
                    m, n, player, path=path, name='player_min_values.png')
                writer.add_image(
                    'player_min_value',
                    skimage.io.imread(
                        os.path.join(path, 'player_min_values.png')))

                _np_plot_wythoff_min_values(
                    m, n, opponent, path=path, name='opp_min_values.png')
                writer.add_image(
                    'opponent_min_value',
                    skimage.io.imread(
                        os.path.join(path, 'opp_min_values.png')))

    # ------------------------------------------------------------------------
    # The end
    if tensorboard:
        writer.close()

    return player, opponent, env