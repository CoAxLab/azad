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
from azad.local_gym.wythoff import locate_cold_moves

from azad.models import Table
from azad.models import DeepTable3
from azad.models import HotCold2
from azad.models import HotCold3
from azad.models import ReplayMemory
from azad.policy import epsilon_greedy
from azad.policy import softmax
# from azad.policy import greedy

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
from azad.util.wythoff import create_bias_board
from azad.util.wythoff import create_all_possible_moves
from azad.util.wythoff import create_cold_board
from azad.util.wythoff import _np_epsilon_greedy
from azad.util.wythoff import _np_expected_value
from azad.util.wythoff import _np_greedy
from azad.util.wythoff import _np_plot_wythoff_max_values
from azad.util.wythoff import plot_cold_board
from azad.util.wythoff import plot_wythoff_board
from azad.util.wythoff import plot_wythoff_expected_values


def wythoff_agent(path,
                  num_trials=10,
                  epsilon=0.1,
                  gamma=0.8,
                  learning_rate=0.1,
                  game='Wythoff10x10',
                  model=None,
                  bias_board=None,
                  tensorboard=False,
                  debug=False,
                  seed=None):
    """Train a Q-agent to play Wythoff's game, using a lookup table."""
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
    m, n, board, moves = peek(env)

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
            move_i = _np_epsilon_greedy(Qs, epsilon=epsilon)

            # Freeze indices so can update Q(s,a) after the opponent
            # moves.
            grad_i = deepcopy(move_i)
            grad_board = deepcopy(board)

            move = moves[move_i]

            # Get Q(s, ....)
            Q = Qs[move_i]

            # Play
            (x, y, board, moves), reward, done, _ = env.step(move)
            board = tuple(flatten_board(board).numpy())

            steps += 1

            if debug:
                print(">>> PLAYER move {}".format(move))

            # ----------------------------------------------------------------
            # Greedy opponent plays?
            if not done:

                # Generate moves
                try:
                    move_i = _np_greedy(model[board])
                except KeyError:
                    move_i = np.random.randint(0, len(moves))

                move = moves[move_i]

                # Play
                (x, y, board, moves), reward, done, _ = env.step(move)
                board = tuple(flatten_board(board).numpy())

                # Count opponent moves
                steps += 1

                # Flip reward
                reward *= -1

                if debug:
                    print(">>> OPPONENT move {}".format(move))

            # ----------------------------------------------------------------
            # Learn!
            # Value the moves
            try:
                max_Q = model[board].max()
            except KeyError:
                model[board] = np.ones(len(moves)) * default_Q
                max_Q = model[board].max()

            next_Q = reward + (gamma * max_Q)

            loss = next_Q - Q
            model[grad_board][grad_i] = Q + (learning_rate * loss)

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


def wythoff_agent_unroll(path,
                         num_trials=10,
                         epsilon=0.1,
                         gamma=0.8,
                         learning_rate=0.1,
                         game='Wythoff10x10',
                         model=None,
                         bias_board=None,
                         tensorboard=False,
                         debug=False,
                         seed=None):
    """Train a Q-agent to play Wythoff's game, using a lookup table."""
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
    m, n, board, moves = peek(env)

    # Init the lookup table
    model = {}
    opponent = {}

    # Run some trials
    for trial in range(num_trials):
        # ----------------------------------------------------------------
        # Re-init
        steps = 1

        x, y, board, moves = env.reset()
        board = tuple(flatten_board(board).numpy())
        if debug:
            print("---------------------------------------")
            print(">>> NEW GAME ({}).".format(trial))
            print(">>> Initial position ({}, {})".format(x, y))
            print(">>> Initial moves {}".format(moves))
            print("---------------------------------------")

        # State vars
        t_states = [
            board,
        ]
        t_all_moves = [moves]
        t_move = []
        t_move_i = []
        t_rewards = []

        done = False
        player_win = False
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
            move_i = _np_epsilon_greedy(Qs, epsilon=epsilon)
            move = moves[move_i]

            # Get Q(s, ....)
            Q = Qs[move_i]

            # Play
            (x, y, board, moves), reward, done, _ = env.step(move)
            board = tuple(flatten_board(board).numpy())

            steps += 1

            if debug:
                print(">>> PLAYER move {}".format(move))

            t_states.append(board)
            t_move.append(move)
            t_all_moves.append(moves)
            t_move_i.append(move_i)
            t_rewards.append(reward)

            if done:
                player_win = True
                t_states.append(board)
                t_move.append(move)
                t_all_moves.append(moves)
                t_move_i.append(move_i)
                t_rewards.append(reward)

            # ----------------------------------------------------------------
            # Greedy opponent plays?
            if not done:

                # Generate moves
                try:
                    move_i = _np_greedy(opponent[board])
                except KeyError:
                    opponent[board] = np.ones(len(moves)) * default_Q
                    move_i = np.random.randint(0, len(moves))

                move = moves[move_i]

                # Play
                (x, y, board, moves), reward, done, _ = env.step(move)
                board = tuple(flatten_board(board).numpy())

                # Count opponent moves
                steps += 1

                if debug:
                    print(">>> OPPONENT move {}".format(move))

                t_states.append(board)
                t_move.append(move)
                t_all_moves.append(moves)
                t_move_i.append(move_i)
                t_rewards.append(reward)

                if done:
                    t_states.append(board)
                    t_move.append(move)
                    t_all_moves.append(moves)
                    t_move_i.append(move_i)
                    t_rewards.append(reward)

        # ----------------------------------------------------------------
        # Learn by unrolling the game
        # Player (model)
        s_idx = np.arange(0, steps - 1, 2)
        for i in s_idx:
            # States and actions
            s = t_states[i]
            next_s = t_states[i + 2]
            m_i = t_move_i[i]

            # Value and reward
            Q = model[s][m_i]

            try:
                max_Q = model[next_s].max()
            except KeyError:
                model[next_s] = np.ones(len(t_all_moves[i])) * default_Q
                max_Q = model[next_s].max()

            if player_win:
                r = t_rewards[i]
            else:
                r = -1 * t_rewards[i + 1]

            # Loss and learn
            next_Q = r + (gamma * max_Q)
            loss = next_Q - Q
            model[s][m_i] = Q + (learning_rate * loss)

        # Opponent
        s_idx = np.arange(1, steps - 1, 2)
        for i in s_idx:
            # States and actions
            s = t_states[i]
            next_s = t_states[i + 2]
            m_i = t_move_i[i]

            # Value and reward
            Q = opponent[s][m_i]

            try:
                max_Q = opponent[next_s].max()
            except KeyError:
                opponent[next_s] = np.ones(len(t_all_moves[i])) * default_Q
                max_Q = opponent[next_s].max()

            if not player_win:
                r = t_rewards[i]
            else:
                r = -1 * t_rewards[i + 1]

            # Loss and learn
            next_Q = r + (gamma * max_Q)
            loss = next_Q - Q
            opponent[s][m_i] = Q + (learning_rate * loss)

        # ----------------------------------------------------------------
        # Update the log
        if debug:
            print(">>> Reward {}; Loss(Q {}, next_Q {}) -> {}".format(
                r, Q, next_Q, loss))

            if done and (r > 0):
                print("*** WIN ***")
            if done and (r < 0):
                print("*** OPPONENT WIN ***")

        if tensorboard and (int(trial) % 50) == 0:
            writer.add_scalar(os.path.join(path, 'reward'), r, trial)
            writer.add_scalar(os.path.join(path, 'Q'), Q, trial)
            writer.add_scalar(os.path.join(path, 'error'), loss, trial)
            writer.add_scalar(os.path.join(path, 'steps'), steps, trial)

            # Optimal ref:
            plot_cold_board(m, n, path=path, name='cold_board.png')
            writer.add_image(
                'cold_positions',
                skimage.io.imread(os.path.join(path, 'cold_board.png')))

            _np_plot_wythoff_max_values(
                m, n, model, path=path, name='player_max_values.png')
            writer.add_image(
                'player',
                skimage.io.imread(
                    os.path.join(path, 'player_max_values.png')))

            _np_plot_wythoff_max_values(
                m, n, opponent, path=path, name='opponent_max_values.png')
            writer.add_image(
                'opponent',
                skimage.io.imread(
                    os.path.join(path, 'opponent_max_values.png')))

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
                     env=None,
                     bias_board=None,
                     tensorboard=False,
                     debug=False,
                     update_every=100,
                     anneal=False,
                     independent_opp=False,
                     save=False,
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

    if tensorboard:
        writer = SummaryWriter(log_dir=path)

    # Crate env
    if env is None:
        env = create_env(game)

    env.seed(seed)
    np.random.seed(seed)
    avg_optim = 0.0

    # ------------------------------------------------------------------------
    # Build a Q agent, and its optimizer
    m, n, board, _ = peek(env)
    all_possible_moves = create_all_possible_moves(m, n)

    # Init the model and top
    if model is None:
        model = Table(m * n, len(all_possible_moves))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if independent_opp:
        opp_model = Table(m * n, len(all_possible_moves))
        opp_optimizer = optim.SGD(opp_model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------------
    # Run some trials
    for trial in range(num_trials):
        # --------------------------------------------------------------------
        # Re-init
        steps = 0
        num_cold = 0
        good = 0
        best = 0

        # Start the game, and process the result
        x, y, board, moves = env.reset()
        board = flatten_board(board)
        moves_index = locate_moves(moves, all_possible_moves)

        if debug:
            print("---------------------------------------")
            print(">>> STUMBLER ({}).".format(trial))
            print(">>> Initial position ({}, {})".format(x, y))
            print(">>> Initial moves {}".format(moves))

        # --------------------------------------------------------------------
        done = False
        while not done:
            # ----------------------------------------------------------------
            # PLAYER

            # Create values
            grad_board = board.clone()
            Qs = model(board)

            # Bias Q?
            if bias_board is not None:
                Qs.add_(flatten_board(bias_board[0:m, 0:n]))

            # Move!
            k = 0.3
            with torch.no_grad():
                if anneal:
                    epsilon_t = epsilon * (1.0 / np.log((trial + np.e)))
                else:
                    epsilon_t = epsilon
                move_i = epsilon_greedy(Qs, epsilon_t, index=moves_index)

                grad_i = deepcopy(move_i)
                move = all_possible_moves[grad_i]

            # Was it a wise move?
            if cold_move_available(x, y, moves):
                num_cold += 1
                if move == locate_closest_cold_move(x, y, moves):
                    best += 1

                if move in locate_cold_moves(x, y, moves):
                    good += 1

            # Get Q(s, a)
            Q = Qs.gather(0, torch.tensor(grad_i))

            if debug:
                print(">>> Possible moves {}".format(moves))
                print(">>> PLAYER move {}".format(move))

            # Play
            (x, y, board, moves), reward, done, _ = env.step(move)
            board = flatten_board(board)
            moves_index = locate_moves(moves, all_possible_moves)

            # Count moves
            steps += 1

            # ----------------------------------------------------------------
            # Greedy opponent plays?
            if done:
                if independent_opp:
                    opp_reward = -1 * reward
            if not done:
                if independent_opp:
                    with torch.no_grad():
                        move_i = epsilon_greedy(
                            opp_model(board), epsilon=0.0, index=moves_index)
                        move = all_possible_moves[move_i]
                else:
                    with torch.no_grad():
                        # move_i = np.random.choice(moves_index)
                        move_i = epsilon_greedy(
                            model(board), epsilon=0.0, index=moves_index)
                        move = all_possible_moves[move_i]

                if debug:
                    print(">>> Possible moves {}".format(moves))
                    print(">>> OPPONENT move {}".format(move))

                # Play
                (x, y, board, moves), reward, done, _ = env.step(move)
                board = flatten_board(board)
                moves_index = locate_moves(moves, all_possible_moves)

                if independent_opp:
                    opp_reward = deepcopy(reward)

                # Flip reward for player update
                reward *= -1

                # Count moves
                # steps += 1

            if independent_opp:
                opp_next_Qs = opp_model(board).detach()
                opp_next_Qs.gather(0, torch.tensor(moves_index))
                opp_max_Q = opp_next_Qs.max()

                opp_next_Q = (opp_reward + (gamma * opp_max_Q))
                opp_loss = F.l1_loss(Q, opp_next_Q)

                opp_optimizer.zero_grad()
                opp_loss.backward(retain_graph=True)
                opp_optimizer.step()

            # ----------------------------------------------------------------
            # Learn!
            next_Qs = model(board).detach()
            next_Qs.gather(0, torch.tensor())
            max_Q = next_Qs.max()

            next_Q = (reward + (gamma * max_Q))
            loss = F.l1_loss(Q, next_Q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update avg performance metric
            if num_cold > 0:
                avg_optim += (best - avg_optim) / (trial + 1)

            # ----------------------------------------------------------------
            if debug:
                print(">>> Reward {}; max_Q {}; Loss(Q {}, next_Q {}) -> {}".
                      format(reward, max_Q, float(Q.detach().numpy()), next_Q,
                             loss.data[0]))
                print(">>> Board grad: {}".format(grad_board))
                print(">>> W grad: {}".format(model.fc1.weight.grad))
                if done and (reward > 0):
                    print("*** WIN ***")
                if done and (reward < 0):
                    print("*** OPPONENT WIN ***")

            if tensorboard and (int(trial) % update_every) == 0:
                writer.add_graph(model, (board, ))

                writer.add_scalar(os.path.join(path, 'reward'), reward, trial)
                writer.add_scalar(os.path.join(path, 'Q'), Q, trial)
                writer.add_scalar(os.path.join(path, 'error'), loss, trial)
                writer.add_scalar(
                    os.path.join(path, 'epsilon_t'), epsilon_t, trial)
                writer.add_scalar(os.path.join(path, 'steps'), steps, trial)
                writer.add_scalar(
                    os.path.join(path, 'avg_optimal'), avg_optim, trial)

                # Optimal ref:
                plot_cold_board(m, n, path=path, name='cold_board.png')
                writer.add_image(
                    'True cold positions',
                    skimage.io.imread(os.path.join(path, 'cold_board.png')))

                # EV:
                plot_wythoff_expected_values(
                    m, n, model, path=path, name='wythoff_expected_values.png')
                writer.add_image(
                    'Max value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_expected_values.png')))

                est_hc_board = estimate_hot_cold(
                    m, n, model, hot_threshold=0.5, cold_threshold=0.0)
                plot_wythoff_board(
                    est_hc_board, path=path, name='est_hc_board.png')
                writer.add_image(
                    'Estimated hot and cold positions',
                    skimage.io.imread(os.path.join(path, 'est_hc_board.png')))

    # ------------------------------------------------------------------------
    # The end
    if tensorboard:
        writer.close()

    if save:
        state = {
            'trial': trial,
            'game': game,
            'epsilon': epsilon,
            'gamma': gamma,
            'learning_rate': learning_rate,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(path + ".pytorch"))

    if debug:
        print(">>> Final W: {}".format(model.fc1.weight))

    return model, env


def wythoff_stumbler_joint(path,
                           num_trials=10,
                           epsilon=0.1,
                           gamma=0.8,
                           learning_rate=0.1,
                           game='Wythoff3x3',
                           env=None,
                           bias_board=None,
                           tensorboard=False,
                           debug=False,
                           update_every=100,
                           anneal=False,
                           save=False,
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

    if tensorboard:
        writer = SummaryWriter(log_dir=path)

    # Crate env
    if env is None:
        env = create_env(game)

    env.seed(seed)
    np.random.seed(seed)
    avg_optim = 0.0

    # ------------------------------------------------------------------------
    # Build a Q agent, and its optimizer
    m, n, board, _ = peek(env)
    all_possible_moves = create_all_possible_moves(m, n)

    # Joint action space
    num_actions = 2 * m * n

    # Init the model and top
    model = Table(m * n, num_actions)
    opp_model = Table(m * n, num_actions)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    opp_optimizer = optim.SGD(opp_model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------------
    # Run some trials
    for trial in range(num_trials):
        # --------------------------------------------------------------------
        # Re-init
        steps = 0
        num_cold = 0
        good = 0
        best = 0

        # Start the game, and process the result
        x, y, board, moves = env.reset()
        board = flatten_board(board)
        moves_index = locate_moves(moves, all_possible_moves)

        if debug:
            print("---------------------------------------")
            print(">>> STUMBLER ({}).".format(trial))
            print(">>> Initial position ({}, {})".format(x, y))
            print(">>> Initial moves {}".format(moves))

        # --------------------------------------------------------------------
        done = False
        while not done:
            # ----------------------------------------------------------------
            # PLAYER

            # Create values
            grad_board = board.clone()
            Qs = model(board)

            # Bias Q?
            if bias_board is not None:
                Qs.add_(flatten_board(bias_board[0:m, 0:n]))

            # Move!
            k = 0.3
            with torch.no_grad():
                if anneal:
                    epsilon_t = epsilon * (1.0 / np.log((trial + np.e)))
                else:
                    epsilon_t = epsilon
                move_i = epsilon_greedy(Qs, epsilon_t, index=moves_index)

                grad_i = deepcopy(move_i)
                move = all_possible_moves[grad_i]

            # Was it a wise move?
            if cold_move_available(x, y, moves):
                num_cold += 1
                if move == locate_closest_cold_move(x, y, moves):
                    best += 1

                if move in locate_cold_moves(x, y, moves):
                    good += 1

            # Get Q(s, a)
            Q = Qs.gather(0, torch.tensor(grad_i))

            if debug:
                print(">>> Possible moves {}".format(moves))
                print(">>> PLAYER move {}".format(move))

            # Play
            (x, y, board, moves), reward, done, _ = env.step(move)
            board = flatten_board(board)
            moves_index = locate_moves(moves, all_possible_moves)

            # Count moves
            steps += 1

            # ----------------------------------------------------------------
            # Greedy opponent plays?
            if done:
                if independent_opp:
                    opp_reward = -1 * reward
            if not done:
                if independent_opp:
                    with torch.no_grad():
                        move_i = epsilon_greedy(
                            opp_model(board), epsilon=0.0, index=moves_index)
                        move = all_possible_moves[move_i]
                else:
                    with torch.no_grad():
                        # move_i = np.random.choice(moves_index)
                        move_i = epsilon_greedy(
                            model(board), epsilon=0.0, index=moves_index)
                        move = all_possible_moves[move_i]

                if debug:
                    print(">>> Possible moves {}".format(moves))
                    print(">>> OPPONENT move {}".format(move))

                # Play
                (x, y, board, moves), reward, done, _ = env.step(move)
                board = flatten_board(board)
                moves_index = locate_moves(moves, all_possible_moves)

                if independent_opp:
                    opp_reward = deepcopy(reward)

                # Flip reward for player update
                reward *= -1

                # Count moves
                # steps += 1

            if independent_opp:
                opp_next_Qs = opp_model(board).detach()
                opp_next_Qs.gather(0, torch.tensor(moves_index))
                opp_max_Q = opp_next_Qs.max()

                opp_next_Q = (opp_reward + (gamma * opp_max_Q))
                opp_loss = F.l1_loss(Q, opp_next_Q)

                opp_optimizer.zero_grad()
                opp_loss.backward(retain_graph=True)
                opp_optimizer.step()

            # ----------------------------------------------------------------
            # Learn!
            next_Qs = model(board).detach()
            next_Qs.gather(0, torch.tensor(moves_index))
            max_Q = next_Qs.max()

            next_Q = (reward + (gamma * max_Q))
            loss = F.l1_loss(Q, next_Q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update avg performance metric
            if num_cold > 0:
                avg_optim += (best - avg_optim) / (trial + 1)

            # ----------------------------------------------------------------
            if debug:
                print(">>> Reward {}; max_Q {}; Loss(Q {}, next_Q {}) -> {}".
                      format(reward, max_Q, float(Q.detach().numpy()), next_Q,
                             loss.data[0]))
                print(">>> Board grad: {}".format(grad_board))
                print(">>> W grad: {}".format(model.fc1.weight.grad))
                if done and (reward > 0):
                    print("*** WIN ***")
                if done and (reward < 0):
                    print("*** OPPONENT WIN ***")

            if tensorboard and (int(trial) % update_every) == 0:
                writer.add_graph(model, (board, ))

                writer.add_scalar(os.path.join(path, 'reward'), reward, trial)
                writer.add_scalar(os.path.join(path, 'Q'), Q, trial)
                writer.add_scalar(os.path.join(path, 'error'), loss, trial)
                writer.add_scalar(
                    os.path.join(path, 'epsilon_t'), epsilon_t, trial)
                writer.add_scalar(os.path.join(path, 'steps'), steps, trial)
                writer.add_scalar(
                    os.path.join(path, 'avg_optimal'), avg_optim, trial)

                # Optimal ref:
                plot_cold_board(m, n, path=path, name='cold_board.png')
                writer.add_image(
                    'True cold positions',
                    skimage.io.imread(os.path.join(path, 'cold_board.png')))

                # EV:
                plot_wythoff_expected_values(
                    m, n, model, path=path, name='wythoff_expected_values.png')
                writer.add_image(
                    'Max value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_expected_values.png')))

                est_hc_board = estimate_hot_cold(
                    m, n, model, hot_threshold=0.5, cold_threshold=0.0)
                plot_wythoff_board(
                    est_hc_board, path=path, name='est_hc_board.png')
                writer.add_image(
                    'Estimated hot and cold positions',
                    skimage.io.imread(os.path.join(path, 'est_hc_board.png')))

    # ------------------------------------------------------------------------
    # The end
    if tensorboard:
        writer.close()

    if save:
        state = {
            'trial': trial,
            'game': game,
            'epsilon': epsilon,
            'gamma': gamma,
            'learning_rate': learning_rate,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(path + ".pytorch"))

    if debug:
        print(">>> Final W: {}".format(model.fc1.weight))

    return model, env


def wythoff_strategist(path,
                       num_trials=1000,
                       num_stumbles=1,
                       num_evals=1,
                       epsilon=0.2,
                       gamma=0.98,
                       cold_threshold=0.0,
                       stumbler_learning_rate=0.1,
                       strategist_learning_rate=0.01,
                       stumbler_game='Wythoff15x15',
                       strategist_game='Wythoff50x50',
                       tensorboard=False,
                       debug=False,
                       save=False,
                       seed=None):
    """A stumbler-strategist netowrk"""

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

    # Create env and find all moves in it
    env = create_env(strategist_game)
    m, n, board, _ = peek(env)
    all_possible_moves = create_all_possible_moves(m, n)

    # Peek at stumbler env
    o, p, _, _ = peek(create_env(stumbler_game))

    # Control randomness
    env.seed(seed)
    np.random.seed(seed)

    # Working mem size
    batch_size = 64

    if tensorboard:
        writer = SummaryWriter(log_dir=path)

    # ------------------------------------------------------------------------
    # Build a Strategist, its memory, and its optimizer
    num_hidden1 = 100
    num_hidden2 = 25
    model = HotCold3(2, num_hidden1=num_hidden1, num_hidden2=num_hidden2)

    optimizer = optim.Adam(model.parameters(), lr=strategist_learning_rate)
    memory = ReplayMemory(500)

    # ------------------------------------------------------------------------
    # Train over trials:

    # Init
    stumbler_model = None
    stumbler_env = None

    influence = 0.0
    bias_board = torch.zeros((m, n), dtype=torch.float)

    # Run
    for trial in range(num_trials):
        # --------------------------------------------------------------------
        stumbler_model, stumbler_env = wythoff_stumbler(
            path,
            num_trials=num_stumbles,
            epsilon=epsilon,
            gamma=gamma,
            game=stumbler_game,
            model=stumbler_model,
            env=stumbler_env,
            bias_board=bias_board * influence,  # None
            learning_rate=stumbler_learning_rate,
            tensorboard=False,
            debug=debug,
            seed=seed)

        if debug:
            print("---------------------------------------")
            print(">>> STRATEGIST ({}).".format(trial))

        # Extract strategic data from the stumber,
        # project it and remember that
        strategic_default_value = 0.0
        strategic_value = estimate_cold(
            o, p, stumbler_model, threshold=cold_threshold)

        # strategic_value = estimate_hot_cold(
        #     o, p, stumbler_model, hot_threshold=0.5, cold_threshold=0.0)

        # ...Into tuples
        s_data = convert_ijv(strategic_value)
        s_data = balance_ijv(s_data, strategic_default_value)
        if s_data is not None:
            for d in s_data:
                memory.push(*d)

        loss = 0.0
        if len(memory) > batch_size:
            # Sample data....
            coords = []
            values = []
            samples = memory.sample(batch_size)

            for c, v in samples:
                coords.append(c)
                values.append(v)

            coords = torch.tensor(
                np.vstack(coords), requires_grad=True, dtype=torch.float)
            values = torch.tensor(
                values, requires_grad=False, dtype=torch.float)

            # Making some preditions,
            predicted_values = model(coords).squeeze()

            # and find their loss.
            loss = F.mse_loss(predicted_values, values)

            # Walk down the hill of righteousness!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if debug:
                print(">>> Coords: {}".format(coords))
                print(">>> Values: {}".format(values))
                print(">>> Predicted values: {}".format(values))
                print(">>> Loss {}".format(loss))

                print(">>> Last win {}".format(win))
                print(">>> Influence {}".format(influence))

        # --------------------------------------------------------------------
        # Use the trained strategist to generate a bias_board,
        bias_board = create_bias_board(m, n, model)

        # Est performance. Count strategist wins.
        win = evaluate_models(
            stumbler_model,
            model,
            stumbler_env,
            env,
            num_eval=num_evals,
            debug=debug)

        # Update the influence and then the bias_board
        if win > 0.5:
            influence += strategist_learning_rate
        else:
            influence -= strategist_learning_rate
        influence = np.clip(influence, 0, 1)

        # --------------------------------------------------------------------
        if tensorboard and (int(trial) % 50) == 0:
            # Timecourse
            writer.add_scalar(
                os.path.join(path, 'stategist_error'), loss, trial)
            writer.add_scalar(os.path.join(path, 'Stategist_wins'), win, trial)
            writer.add_scalar(
                os.path.join(path, 'Stategist_influence'), influence, trial)

            # Boards
            plot_wythoff_expected_values(
                o, p, stumbler_model, vmin=-2, vmax=2, path=path)
            writer.add_image(
                'Stumbler_expected_value',
                skimage.io.imread(
                    os.path.join(path, 'wythoff_expected_values.png')))

            est_hc_board = estimate_hot_cold(
                o, p, stumbler_model, hot_threshold=0.5, cold_threshold=0.0)
            plot_wythoff_board(
                est_hc_board, path=path, name='est_hc_board.png')
            writer.add_image(
                'Stumbler_hot_cold',
                skimage.io.imread(os.path.join(path, 'est_hc_board.png')))

            plot_wythoff_board(
                bias_board, vmin=-1, vmax=0, path=path, name='bias_board.png')
            writer.add_image(
                'Strategist learning',
                skimage.io.imread(os.path.join(path, 'bias_board.png')))

    # ------------------------------------------------------------------------
    # The end
    if tensorboard:
        writer.close()

    if save:
        state = {
            'trial': trial,
            'epsilon': epsilon,
            'gamma': gamma,
            'num_trials': num_trials,
            'num_stumbles': num_stumbles,
            'num_evals': num_evals,
            'influence': influence,
            'stumbler_game': stumbler_game,
            'strategist_game': strategist_game,
            'cold_threshold': cold_threshold,
            'stumbler_learning_rate': stumbler_learning_rate,
            'stumbler_state_dict': stumbler_model.state_dict(),
            'strategist_learning_rate': strategist_learning_rate,
            'strategist_state_dict': model.state_dict(),
            'strategist_optimizer': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(path, "model_state.pytorch"))

        # Save final images
        plot_wythoff_expected_values(
            o, p, stumbler_model, vmin=-2, vmax=2, path=path)

        est_hc_board = estimate_hot_cold(
            o, p, stumbler_model, hot_threshold=0.5, cold_threshold=0.0)
        plot_wythoff_board(est_hc_board, path=path, name='est_hc_board.png')

        plot_wythoff_board(
            bias_board, vmin=-1, vmax=0, path=path, name='bias_board.png')

    return (model, env, influence)


def wythoff_optimal(path,
                    num_trials=1000,
                    learning_rate=0.01,
                    num_hidden1=100,
                    num_hidden2=25,
                    stumbler_game='Wythoff10x10',
                    strategist_game='Wythoff50x50',
                    tensorboard=False,
                    debug=False,
                    seed=None):
    """A minimal example."""

    # ------------------------------------------------------------------------
    # Setup
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    m, n, board, _ = peek(create_env(strategist_game))
    o, p, _, _ = peek(create_env(stumbler_game))

    if debug:
        print(">>> TRANING AN OPTIMAL STRATEGIST.")
        print(">>> Train board {}".format(o, p))
        print(">>> Test board {}".format(m, n))

    # Log setup
    if tensorboard:
        writer = SummaryWriter(log_dir=path)

    # Seeding...
    np.random.seed(seed)

    # Train params
    strategic_default_value = 0.0
    batch_size = 64

    # ------------------------------------------------------------------------
    # Build a Strategist, its memory, and its optimizer

    # Create a model, of the right size.
    # model = HotCold2(2, num_hidden1=num_hidden1)
    model = HotCold3(2, num_hidden1=num_hidden1, num_hidden2=num_hidden2)

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

        loss = 0.0
        if len(memory) > batch_size:
            # Sample data....
            coords = []
            values = []
            samples = memory.sample(batch_size)

            for c, v in samples:
                coords.append(c)
                values.append(v)

            coords = torch.tensor(
                np.vstack(coords), requires_grad=True, dtype=torch.float)
            values = torch.tensor(
                values, requires_grad=False, dtype=torch.float)

            # Making some preditions,
            predicted_values = model(coords).squeeze()

            # and find their loss.
            loss = F.mse_loss(predicted_values, values)

            # Walk down the hill of righteousness!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if debug:
                print(">>> Coords: {}".format(coords))
                print(">>> Values: {}".format(values))
                print(">>> Predicted values: {}".format(values))
                print(">>> Loss {}".format(loss))

        # Use the trained strategist to generate a bias_board,
        bias_board = create_bias_board(m, n, model)

        if tensorboard and (int(trial) % 50) == 0:
            writer.add_scalar(os.path.join(path, 'error'), loss, trial)

            plot_wythoff_board(
                strategic_value,
                vmin=0,
                vmax=1,
                path=path,
                name='strategy_board.png')
            writer.add_image(
                'Training board',
                skimage.io.imread(os.path.join(path, 'strategy_board.png')))

            plot_wythoff_board(
                bias_board, vmin=0, vmax=1, path=path, name='bias_board.png')
            writer.add_image(
                'Testing board',
                skimage.io.imread(os.path.join(path, 'bias_board.png')))

    # The end
    if tensorboard:
        writer.close()

    return model, env,
