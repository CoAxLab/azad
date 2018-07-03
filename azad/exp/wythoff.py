import os, csv
import sys

import errno
import pudb

from collections import defaultdict

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

from azad.models import LinQN1
from azad.models import HotCold2
from azad.models import HotCold3
from azad.models import ReplayMemory
from azad.policy import epsilon_greedy
from azad.policy import greedy
from azad.util.wythoff import *


def wythoff_agent(path,
                  num_trials=10,
                  epsilon=0.1,
                  gamma=0.8,
                  learning_rate=0.1,
                  game='Wythoff3x3',
                  model=None,
                  bias_board=None,
                  tensorboard=False,
                  debug=False,
                  seed=None):
    """Train a Q-agent to play Wythoff's game, using SGD."""
    # ------------------------------------------------------------------------
    # Setup

    # Create path
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # Do tensorboard?
    if tensorboard:
        writer = SummaryWriter(log_dir=path)

    # Crate env
    env = create_env(game)

    # Control randomness
    env.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------------
    # Build a Q agent, and its optimizer
    default_Q = 0.0
    m, n, board, moves = peak(env)

    # Init the lookup table
    model = {}

    # Run some trials
    for trial in range(num_trials):
        # ----------------------------------------------------------------
        # Re-init
        steps = 0

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

            # Get all values
            try:
                Qs = model[board]
            except KeyError:
                model[board] = np.ones(len(moves)) * default_Q
                Qs = model[board]

            # Move!
            player_index = _np_epsilon_greedy(Qs, epsilon=epsilon)
            player_move = moves[player_index]

            (x, y, next_board, moves), reward, done, _ = env.step(player_move)
            next_board = tuple(flatten_board(next_board).numpy())

            steps += 1

            if debug:
                print(">>> PLAYER move {}".format(player_move))

            # ----------------------------------------------------------------
            # Greedy opponent plays?
            if not done:

                # Generate moves
                try:
                    opponent_index = _np_greedy(model[next_board])
                except KeyError:
                    opponent_index = np.random.randint(0, len(moves))

                opponent_move = moves[opponent_index]

                (x, y, next_board,
                 moves), reward, done, _ = env.step(opponent_move)
                next_board = tuple(flatten_board(next_board).numpy())

                reward *= -1

                if debug:
                    print(">>> OPPONENT move {}".format(opponent_move))

            # ----------------------------------------------------------------
            # Learn!
            # Value the moves
            Q = Qs[player_index]
            try:
                max_Q = model[next_board].max()
            except KeyError:
                max_Q = default_Q

            next_Q = reward + (gamma * max_Q)
            loss = next_Q - Q
            model[board][player_index] = Q + (learning_rate * loss)

            # ----------------------------------------------------------------
            # Shuffle state
            board = next_board

            # ----------------------------------------------------------------
            # if tensorboard and (int(trial) % 50) == 0:
            if debug:
                print(">>> Reward {}; Loss(Q {}, next_Q {}) -> {}".format(
                    reward, Q, next_Q, loss))

                if done and (reward > 0):
                    print("*** WIN ***")
                if done and (reward < 0):
                    print("*** OPPONENT WIN ***")

            if tensorboard and (int(trial) % 50) == 0:
                writer.add_scalar(os.path.join(path, 'reward'), reward, trial)
                writer.add_scalar(os.path.join(path, 'Q'), Q, trial)
                writer.add_scalar(
                    os.path.join(path, 'error'), loss.data[0], trial)
                writer.add_scalar(os.path.join(path, 'steps'), steps, trial)

                # Optimal ref:
                plot_cold_board(m, n, path=path, name='cold_board.png')
                writer.add_image(
                    'cold_positions',
                    skimage.io.imread(os.path.join(path, 'cold_board.png')))

                # EV:
                _np_plot_wythoff_expected_values(
                    m, n, model, path=path, name='wythoff_expected_values.png')
                writer.add_image(
                    'expected_value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_expected_values.png')))

                _np_plot_wythoff_min_values(
                    m, n, model, path=path, name='wythoff_min_values.png')
                writer.add_image(
                    'min_value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_min_values.png')))

                _np_plot_wythoff_max_values(
                    m, n, model, path=path, name='wythoff_max_values.png')
                writer.add_image(
                    'max_value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_max_values.png')))

    # ------------------------------------------------------------------------
    # The end
    if tensorboard:
        writer.close()

    return model, env


def wythoff_stumbler(path,
                     num_trials=10,
                     epsilon=0.1,
                     gamma=0.8,
                     learning_rate=0.1,
                     game='Wythoff3x3',
                     model=None,
                     bias_board=None,
                     tensorboard=False,
                     seed=None):
    """Train a NN-based Q-agent to play Wythoff's game, using SGD."""
    # -------------------------------------------
    # Setup

    # Create path
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # Do tensorboard?
    if tensorboard:
        writer = SummaryWriter(log_dir=path)

    # Crate env
    env = create_env(game)

    # Control randomness
    env.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------------
    # Build a Q agent, and its optimizer
    default_Q = 0.0
    m, n, board, moves = peak(env)

    # Init the model
    if model is None:
        model = LinQN1(m * n, len(all_actions))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # -------------------------------------------
    # Run some trials
    for trial in range(num_trials):
        # ----------------------------------------------------------------
        # Re-init
        steps = 0

        x, y, board, moves = env.reset()
        board = tuple(flatten_board(board).numpy())

        if debug:
            print("---------------------------------------")
            print(">>> NEW GAME ({}).".format(trial))
            print(">>> Initial position ({}, {})".format(x, y))
            print(">>> Initial moves {}".format(moves))
            print("---------------------------------------")

        # --------------------------------------------------------------------
        done = False
        while done:
            # ----------------------------------------------------------------
            # PLAYER

            # Get all the values
            Qs = model(board)

            # Bias Q?
            # Qs.add_(create_Q_bias(x, y, bias_board, Qs, possible_index))

            # Move!
            with torch.no_grad():
                player_index = epsilon_greedy(
                    Qs, epsilon, index=possible_index)
                player_move = moves[player_index]

            (x, y, next_board, moves), reward, done, _ = env.step(player_move)
            next_board = tuple(flatten_board(next_board).numpy())

            steps += 1

            if debug:
                print(">>> PLAYER move {}".format(player_move))

            # ----------------------------------------------------------------
            # Greedy opponent plays?
            if not done:
                with torch.no_grad():
                    opponent_index = greedy(model(next_board))
                    opponent_move = moves[opponent_index]

                (x, y, next_board,
                 moves), reward, done, _ = env.step(opponent_move)
                next_board = flatten_board(next_board)

                reward *= -1

                if debug:
                    print(">>> OPPONENT move {}".format(opponent_move))

            # ----------------------------------------------------------------
            # Learn!
            Q = Qs.gather(0, torch.tensor(player_index))

            max_Q = model(next_board).detach().max()
            next_Q = reward + (gamma * max_Q)
            loss = F.smooth_l1_loss(Q, next_Q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ----------------------------------------------------------------
            # Prep for next iteration
            board = next_board

            # ----------------------------------------------------------------
            if debug:
                print(">>> Reward {}; Loss(Q {}, next_Q {}) -> {}".format(
                    reward, Q, next_Q, loss))

                if done and (reward > 0):
                    print("*** WIN ***")
                if done and (reward < 0):
                    print("*** OPPONENT WIN ***")

            if tensorboard and (int(trial) % 50) == 0:
                writer.add_scalar(os.path.join(path, 'reward'), reward, trial)
                writer.add_scalar(os.path.join(path, 'Q'), Q, trial)
                writer.add_scalar(
                    os.path.join(path, 'error'), loss.data[0], trial)
                writer.add_scalar(os.path.join(path, 'steps'), steps, trial)

                # Optimal ref:
                plot_cold_board(m, n, path=path, name='cold_board.png')
                writer.add_image(
                    'cold_positions',
                    skimage.io.imread(os.path.join(path, 'cold_board.png')))

                # EV:
                _np_plot_wythoff_expected_values(
                    m, n, model, path=path, name='wythoff_expected_values.png')
                writer.add_image(
                    'expected_value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_expected_values.png')))

                _np_plot_wythoff_min_values(
                    m, n, model, path=path, name='wythoff_min_values.png')
                writer.add_image(
                    'min_value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_min_values.png')))

                _np_plot_wythoff_max_values(
                    m, n, model, path=path, name='wythoff_max_values.png')
                writer.add_image(
                    'max_value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_max_values.png')))

    # ------------------------------------------------------------------------
    # The end
    if tensorboard:
        writer.close()

    return model, env


def wythoff_strategist(
        path,
        num_trials=1000,
        num_stumbles=100,
        epsilon=0.1,
        gamma=0.8,
        delta=0.1,
        learning_rate=0.1,
        stumbler_game='Wythoff15x15',
        strategist_game='Wythoff50x50',
        num_hidden1=15,
        #    num_hidden1=25,
        #    num_hidden2=100,
        log=False,
        seed=None):

    # -------------------------------------------
    # Setup
    env = create_env(strategist_game)
    possible_actions = [(-1, 0), (0, -1), (-1, -1)]

    # Working mem size
    batch_size = 12

    # Control randomness
    env.seed(seed)
    np.random.seed(seed)

    if log:
        writer = SummaryWriter(log_dir=path)

    # -------------------------------------------
    # Build a Strategist, its memory, and its optimizer

    # Board sizes....
    m, n, board = peak(env)
    o, p, _ = peak(create_env(stumbler_game))

    # Def the strategist model, opt, and mem
    # model = HotCold3(2, num_hidden1=num_hidden1, num_hidden2=num_hidden2)
    model = HotCold2(2, num_hidden1=num_hidden1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    memory = ReplayMemory(10000)

    # -------------------------------------------
    wins = []
    stumbler_model = None
    bias_board = None
    for trial in range(num_trials):

        stumbler_model, stumbler_env = wythoff_stumbler(
            path,
            num_trials=num_stumbles,
            epsilon=epsilon,
            gamma=gamma,
            game=stumbler_game,
            model=stumbler_model,
            bias_board=bias_board,
            learning_rate=learning_rate)

        # Extract strategic data from the stumber,
        # project it and remember that
        strategic_default_value = 0.0
        strategic_value = -1 * create_cold_board(o, p)

        # strategic_default_value = 0.5
        # strategic_value = estimate_alp_hot_cold(
        #     o, p, stumbler_model, conf=0.05, default=strategic_default_value)
        strategic_value = pad_board(m, n, strategic_value,
                                    strategic_default_value)

        # ...Into tuples
        s_data = convert_ijv(strategic_value)
        s_data = balance_ijv(s_data, strategic_default_value)

        for d in s_data:
            memory.push(*d)

        # Sample data....
        coords = []
        values = []
        samples = memory.sample(batch_size)

        for c, v in samples:
            coords.append(c)
            values.append(v)

        coords = torch.tensor(
            np.vstack(coords), requires_grad=True, dtype=torch.float)
        values = torch.tensor(values, requires_grad=False, dtype=torch.float)

        # Making some preditions,
        predicted_values = model(coords).squeeze()

        # and find their loss.
        loss = F.mse_loss(predicted_values, values)

        # Walk down the hill of righteousness!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Use the trained strategist to generate a bias_board,
        bias_board = delta * estimate_strategic_value(m, n, model)
        # bias_board = delta * strategic_value

        # Compare strategist and stumbler. Count strategist wins.
        win = evaluate_models(stumbler_model, model, stumbler_env, env)

        if log:
            writer.add_scalar(
                os.path.join(path, 'stategist_error'), loss.data[0], trial)

            writer.add_scalar(os.path.join(path, 'stategist_wins'), win, trial)

            plot_wythoff_expected_values(
                o, p, stumbler_model, vmin=-3, vmax=3, path=path)
            writer.add_image(
                'expected_value',
                skimage.io.imread(
                    os.path.join(path, 'wythoff_expected_values.png')))

            plot_wythoff_board(
                strategic_value, path=path, name='strategy_board.png')
            writer.add_image(
                'strategic_value',
                skimage.io.imread(os.path.join(path, 'strategy_board.png')))

            plot_wythoff_board(bias_board, path=path, name='bias_board.png')
            writer.add_image(
                'strategist_model',
                skimage.io.imread(os.path.join(path, 'bias_board.png')))

            plot_q_action_values(
                o,
                p,
                len(possible_actions),
                stumbler_model,
                possible_actions=possible_actions,
                path=path,
                name='q_action_values.png')

            writer.add_image(
                'q_action_values',
                skimage.io.imread(os.path.join(path, 'q_action_values.png')))

    # The end
    if log:
        writer.close()

    return model, stumbler_model, env, stumbler_env, wins


def wythoff_optimal(
        path,
        num_trials=1000,
        learning_rate=0.01,
        num_hidden1=15,
        # num_hidden1=100,
        # num_hidden2=25,
        stumbler_game='Wythoff50x50',
        strategist_game='Wythoff50x50',
        log=False,
        seed=None):
    """A minimal example."""

    # -------------------------------------------
    # Setup
    # -------------------------------------------
    # sCreate path
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    env = gym.make('{}-v0'.format(strategist_game))
    env = wrappers.Monitor(
        env, './tmp/{}-v0-1'.format(strategist_game), force=True)

    possible_actions = [(-1, 0), (0, -1), (-1, -1)]

    # Train params
    strategic_default_value = 0.0
    batch_size = 12

    # -------------------------------------------
    # Log setup
    if log:
        writer = SummaryWriter(log_dir=path)

    # -------------------------------------------
    # Seeding...
    env.seed(seed)
    np.random.seed(seed)

    # -------------------------------------------
    # Build a Strategist, its memory, and its optimizer

    # How big are the boards?
    m, n, board = peak(env)
    o, p, _ = peak(env)

    # Create a model, of the right size.
    model = HotCold2(2, num_hidden1=num_hidden1)
    # model = HotCold3(2, num_hidden1=num_hidden1, num_hidden2=num_hidden2)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    memory = ReplayMemory(10000)

    # Run learning trials. The 'stumbler' is just the opt
    # cold board
    for trial in range(num_trials):
        # The cold spots are '1' everythig else is '0'
        strategic_value = create_cold_board(o, p)

        # ...Into tuples
        s_data = convert_ijv(strategic_value)
        s_data = balance_ijv(s_data, strategic_default_value)

        for d in s_data:
            memory.push(*d)

        # Sample data....
        coords = []
        values = []
        samples = memory.sample(batch_size)

        for c, v in samples:
            coords.append(c)
            values.append(v)

        coords = torch.tensor(
            np.vstack(coords), requires_grad=True, dtype=torch.float)
        values = torch.tensor(values, requires_grad=False, dtype=torch.float)

        # Making some preditions,
        predicted_values = model(coords).squeeze()

        # and find their loss.
        loss = F.mse_loss(predicted_values, values)

        # Walk down the hill of righteousness!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Use the trained strategist to generate a bias_board,
        bias_board = estimate_strategic_value(m, n, model)

        if log:
            writer.add_scalar(os.path.join(path, 'error'), loss.data[0], trial)

            plot_wythoff_board(
                strategic_value,
                vmin=0,
                vmax=1,
                path=path,
                name='strategy_board.png')
            writer.add_image(
                'Expected value board',
                skimage.io.imread(os.path.join(path, 'strategy_board.png')))

            plot_wythoff_board(
                bias_board, vmin=0, vmax=1, path=path, name='bias_board.png')
            writer.add_image(
                'Strategist learning',
                skimage.io.imread(os.path.join(path, 'bias_board.png')))

    # The end
    if log:
        writer.close()

    return model, env
