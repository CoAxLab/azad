import os, csv
import sys

import errno
import pudb

from collections import defaultdict
from copy import deepcopy

import torch
import torch as th
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


def wythoff_stumbler_strategist(num_episodes=10,
                                num_stumbles=1000,
                                stumbler_game='Wythoff10x10',
                                learning_rate_stumbler=0.1,
                                epsilon=0.5,
                                anneal=True,
                                gamma=1.0,
                                num_strategies=1000,
                                strategist_game='Wythoff50x50',
                                learning_rate_strategist=0.01,
                                cold_threshold=0.0,
                                hot_threshold=0.5,
                                hot_value=1,
                                cold_value=-1,
                                num_eval=1,
                                tensorboard=None,
                                update_every=5,
                                seed=None,
                                save=False,
                                debug=False):
    """Learn Wythoff's with a stumbler-strategist network"""

    # -----------------------------------------------------------------------
    # Init

    # Game sizes
    m, n, _, _ = peek(create_env(strategist_game))
    o, p, _, _ = peek(create_env(stumbler_game))

    # Agents, etc
    player = None
    opponent = None
    strategist = None
    bias_board = None

    influence = 0.0
    score_a = 0.0
    score_b = 0.0
    # ------------------------------------------------------------------------
    for episode in range(num_episodes):
        # Stumbler
        (player, opponent), (score_a, _) = wythoff_stumbler(
            num_episodes=num_stumbles,
            game=stumbler_game,
            epsilon=epsilon,
            anneal=anneal,
            gamma=gamma,
            learning_rate=learning_rate_stumbler,
            model=player,
            opponent=opponent,
            bias_board=bias_board,
            influence=influence,
            score=score_a,
            tensorboard=tensorboard,
            update_every=update_every,
            initial=episode * num_stumbles,
            debug=debug,
            seed=seed)

        # Strategist
        strategist, score_b = wythoff_strategist(
            player,
            stumbler_game,
            num_episodes=num_strategies,
            game=strategist_game,
            model=strategist,
            score=score_b,
            cold_threshold=cold_threshold,
            hot_threshold=hot_threshold,
            learning_rate=learning_rate_strategist,
            tensorboard=tensorboard,
            update_every=update_every,
            hot_value=hot_value,
            cold_value=cold_value,
            initial=episode * num_strategies,
            debug=debug,
            seed=seed)

        # --------------------------------------------------------------------
        # Use the trained strategist to generate a bias_board,
        bias_board = create_bias_board(m, n, strategist)

        # Est performance. Count strategist wins.
        win = evaluate_models(
            player,
            strategist,
            stumbler_game,
            strategist_game,
            num_episodes=num_eval,
            debug=debug)

        # Update the influence and then the bias_board
        if win > 0.5:
            influence += learning_rate_strategist
        else:
            influence -= learning_rate_strategist
        influence = np.clip(influence, 0, 1)

        # ------------------------------------------------------------------------
        if tensorboard is not None:
            try:
                os.makedirs(tensorboard)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

            writer = SummaryWriter(log_dir=tensorboard)
            writer.add_scalar(
                os.path.join(tensorboard, 'stategist_influence'), influence,
                episode)
            writer.add_scalar(
                os.path.join(tensorboard, 'stategist_score'), score_b, episode)

            plot_wythoff_board(
                bias_board,
                vmin=-1.5,
                vmax=1.5,
                path=tensorboard,
                height=10,
                width=15,
                name='bias_board.png')
            writer.add_image(
                'strategist_bias_board',
                skimage.io.imread(os.path.join(tensorboard, 'bias_board.png')))
            writer.close()

        # --------------------------------------------------------------------
        if save and (int(episode) % update_every) == 0:
            state = {
                'episode': episode,
                'epsilon': epsilon,
                'anneal': anneal,
                'gamma': gamma,
                'num_episodes': num_episodes,
                'num_stumbles': num_stumbles,
                'num_strategies': num_strategies,
                'influence': influence,
                'stumbler_score': score_a,
                'strategist_score': score_b,
                'stumbler_game': stumbler_game,
                'strategist_game': strategist_game,
                'cold_threshold': cold_threshold,
                'hot_threshold': hot_threshold,
                'learning_rate_stumbler': learning_rate_stumbler,
                'learning_rate_strategist': learning_rate_strategist,
                'strategist_state_dict': strategist.state_dict(),
                'stumbler_player_dict': player,
                'stumbler_opponent_dict': opponent
            }
            torch.save(state, save)

    return (player, opponent, strategist), (score_a, score_a, score_b)


def wythoff_stumbler(num_episodes=10,
                     epsilon=0.1,
                     gamma=0.8,
                     learning_rate=0.1,
                     game='Wythoff10x10',
                     model=None,
                     opponent=None,
                     anneal=False,
                     bias_board=None,
                     influence=0.0,
                     score=0.0,
                     tensorboard=None,
                     update_every=5,
                     initial=0,
                     debug=False,
                     seed=None):
    """Learn to play Wythoff's w/ e-greedy random exploration.
    
    Note: Learning is based on a player-opponent joint action formalism 
    and tabular Q-learning.
    """
    # ------------------------------------------------------------------------
    # Setup
    if tensorboard is not None:
        try:
            os.makedirs(tensorboard)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        writer = SummaryWriter(log_dir=tensorboard)

    # Create env
    env = create_env(game)
    env.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------------
    # Build a Q agent, and its optimizer
    default_Q = 0.0
    m, n, board, available = peek(env)

    # Init the lookup tables?
    if model is None:
        model = {}
    if opponent is None:
        opponent = {}

    # ------------------------------------------------------------------------
    for episode in range(initial, initial + num_episodes):
        # Re-init
        steps = 1

        x, y, board, available = env.reset()
        board = tuple(flatten_board(board))
        if debug:
            print("---------------------------------------")
            print(">>> NEW GAME ({}).".format(episode))
            print(">>> Initial position ({}, {})".format(x, y))
            print(">>> Initial moves {}".format(available))
            print("---------------------------------------")

        t_state = [
            board,
        ]
        t_available = [available]
        t_move = []
        t_move_i = []
        t_reward = []

        # -------------------------------------------------------------------
        # Anneal epsilon?
        if anneal:
            epsilon_e = epsilon * (1.0 / np.log((episode + np.e)))
        else:
            epsilon_e = episode

        # -------------------------------------------------------------------
        # Play a game!
        done = False
        player_win = False
        while not done:
            # PLAYER CHOOSES A MOVE
            try:
                Qs_episode = add_bias_board(model[board], available,
                                            bias_board, influence)
                move_i = epsilon_greedy(
                    Qs_episode, epsilon=epsilon_e, mode='numpy')
            except KeyError:
                model[board] = np.ones(len(available)) * default_Q
                move_i = np.random.randint(0, len(available))

            move = available[move_i]

            # Analyze it...
            best = 0.0
            if cold_move_available(x, y, available):
                if move in locate_cold_moves(x, y, available):
                    best = 1.0
                score += (best - score) / (episode + 1)

            # PLAY THE MOVE
            (x, y, board, available), reward, done, _ = env.step(move)
            board = tuple(flatten_board(board))
            steps += 1

            # Log....
            if debug:
                print(">>> PLAYER move {}".format(move))

            t_state.append(board)
            t_move.append(move)
            t_available.append(available)
            t_move_i.append(move_i)
            t_reward.append(reward)

            if done:
                player_win = True
                t_state.append(board)
                t_move.append(move)
                t_available.append(available)
                t_move_i.append(move_i)
                t_reward.append(reward)

            # ----------------------------------------------------------------
            if not done:
                # OPPONENT CHOOSES A MOVE
                try:
                    Qs_episode = add_bias_board(opponent[board], available,
                                                bias_board, influence)
                    move_i = epsilon_greedy(
                        Qs_episode, epsilon=epsilon_e, mode='numpy')
                except KeyError:
                    opponent[board] = np.ones(len(available)) * default_Q
                    move_i = np.random.randint(0, len(available))

                move = available[move_i]

                # PLAY THE MOVE
                (x, y, board, available), reward, done, _ = env.step(move)
                board = tuple(flatten_board(board))
                steps += 1

                # Log....
                if debug:
                    print(">>> OPPONENT move {}".format(move))

                t_state.append(board)
                t_move.append(move)
                t_available.append(available)
                t_move_i.append(move_i)
                t_reward.append(reward)

                if done:
                    t_state.append(board)
                    t_move.append(move)
                    t_available.append(available)
                    t_move_i.append(move_i)
                    t_reward.append(reward)

        # ----------------------------------------------------------------
        # Learn by unrolling the game...

        # PLAYER (model)
        s_idx = np.arange(0, steps - 1, 2)
        for i in s_idx:
            # States and actions
            s = t_state[i]
            next_s = t_state[i + 2]
            m_i = t_move_i[i]

            # Value and reward
            Q = model[s][m_i]

            try:
                max_Q = model[next_s].max()
            except KeyError:
                model[next_s] = np.ones(len(t_available[i])) * default_Q
                max_Q = model[next_s].max()

            if player_win:
                r = t_reward[i]
            else:
                r = -1 * t_reward[i + 1]

            # Loss and learn
            next_Q = r + (gamma * max_Q)
            loss = next_Q - Q
            model[s][m_i] = Q + (learning_rate * loss)

        # OPPONENT
        s_idx = np.arange(1, steps - 1, 2)
        for i in s_idx:
            # States and actions
            s = t_state[i]
            next_s = t_state[i + 2]
            m_i = t_move_i[i]

            # Value and reward
            Q = opponent[s][m_i]

            try:
                max_Q = opponent[next_s].max()
            except KeyError:
                opponent[next_s] = np.ones(len(t_available[i])) * default_Q
                max_Q = opponent[next_s].max()

            if not player_win:
                r = t_reward[i]
            else:
                r = -1 * t_reward[i + 1]

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

        if tensorboard and (int(episode) % update_every) == 0:
            writer.add_scalar(os.path.join(tensorboard, 'reward'), r, episode)
            writer.add_scalar(os.path.join(tensorboard, 'Q'), Q, episode)
            writer.add_scalar(
                os.path.join(tensorboard, 'error'), loss, episode)
            writer.add_scalar(
                os.path.join(tensorboard, 'steps'), steps, episode)
            writer.add_scalar(
                os.path.join(tensorboard, 'stumbler_score'), score, episode)
            writer.add_scalar(
                os.path.join(tensorboard, 'epsilon'), epsilon_e, episode)

            # Cold ref:
            cold = create_cold_board(m, n)
            plot_wythoff_board(
                cold, vmin=0, vmax=1, path=tensorboard, name='cold_board.png')
            writer.add_image(
                'cold_positions',
                skimage.io.imread(os.path.join(tensorboard, 'cold_board.png')))

            # Agent max(Q) boards
            values = expected_value(m, n, model)
            plot_wythoff_board(
                values, path=tensorboard, name='player_max_values.png')
            writer.add_image(
                'player',
                skimage.io.imread(
                    os.path.join(tensorboard, 'player_max_values.png')))

            values = expected_value(m, n, opponent)
            plot_wythoff_board(
                values, path=tensorboard, name='opponent_max_values.png')
            writer.add_image(
                'opponent',
                skimage.io.imread(
                    os.path.join(tensorboard, 'opponent_max_values.png')))

    # ------------------------------------------------------------------------
    # The end
    if tensorboard is not None:
        writer.close()

    return (model, opponent), (score, score)


def wythoff_strategist(stumbler_model,
                       stumbler_game,
                       num_episodes=1000,
                       cold_threshold=0.0,
                       hot_threshold=0.5,
                       hot_value=1,
                       cold_value=-1,
                       learning_rate=0.01,
                       game='Wythoff50x50',
                       model=None,
                       initial=0,
                       score=0.0,
                       tensorboard=None,
                       stumbler_mode='numpy',
                       update_every=50,
                       num_eval=1,
                       debug=False,
                       seed=None):
    """Learn a generalizable strategy for Wythoffs game"""

    # ------------------------------------------------------------------------
    # Setup
    if tensorboard is not None:
        try:
            os.makedirs(tensorboard)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        writer = SummaryWriter(log_dir=tensorboard)

    # Create env and find all moves in it
    env = create_env(game)
    env.seed(seed)
    np.random.seed(seed)

    m, n, board, _ = peek(env)
    all_possible_moves = create_all_possible_moves(m, n)

    # Peek at stumbler env
    o, p, _, _ = peek(create_env(stumbler_game))

    # Init the strategist net
    if model is None:
        num_hidden1 = 100
        num_hidden2 = 25
        model = HotCold3(2, num_hidden1=num_hidden1, num_hidden2=num_hidden2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------------------------------------------------
    # Extract strategic data from the stumbler
    strategic_default_value = 0.0
    strategic_value = estimate_hot_cold(
        o,
        p,
        stumbler_model,
        hot_threshold=hot_threshold,
        cold_threshold=cold_threshold,
        hot_value=hot_value,
        cold_value=cold_value,
        default_value=strategic_default_value)

    # Convert format
    s_data = convert_ijv(strategic_value)
    # s_data = balance_ijv(s_data, cold_value)

    # Sanity?
    if s_data is None:
        return model, None

    # Define a memory to sample
    memory = ReplayMemory(len(s_data))
    batch_size = len(s_data)
    for d in s_data:
        memory.push(*d)

    # -----------------------------------------------------------------------
    # Sample the memory to teach the strategist
    bias_board = None
    for episode in range(initial, initial + num_episodes):
        loss = 0.0

        if debug:
            print("---------------------------------------")
            print(">>> STRATEGIST ({}).".format(episode))

        coords = []
        values = []
        for c, v in memory.sample(batch_size):
            coords.append(c)
            values.append(v)
        coords = torch.tensor(
            np.vstack(coords), requires_grad=True, dtype=torch.float)
        values = torch.tensor(values, requires_grad=False, dtype=torch.float)

        # Making some preditions, ...
        predicted_values = model(coords).squeeze()

        # and learn from them
        loss = F.mse_loss(predicted_values, values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --------------------------------------------------------------------
        if debug:
            print(">>> Coords: {}".format(coords))
            print(">>> Values: {}".format(values))
            print(">>> Predicted values: {}".format(values))
            print(">>> Loss {}".format(loss))

            print(">>> Last win {}".format(win))

        if tensorboard and (int(episode) % update_every) == 0:
            # Timecourse
            writer.add_scalar(
                os.path.join(tensorboard, 'stategist_error'), loss, episode)

    # Score the model:
    with th.no_grad():
        pred = create_bias_board(m, n, model, default=0.0).numpy()
        cold = create_cold_board(m, n, default=hot_value)
        mae = np.mean(np.abs(pred - cold))
        score += (mae - score) / (episode + 1)

    return (model), (score)


# def wythoff_optimal(path,
#                     num_episodes=1000,
#                     learning_rate=0.01,
#                     num_hidden1=100,
#                     num_hidden2=25,
#                     stumbler_game='Wythoff10x10',
#                     strategist_game='Wythoff50x50',
#                     tensorboard=False,
#                     debug=False,
#                     seed=None):
#     """An optimal stumbler teaches the strategist."""

#     # ------------------------------------------------------------------------
#     # Setup
#     try:
#         os.makedirs(path)
#     except OSError as exception:
#         if exception.errno != errno.EEXIST:
#             raise

#     m, n, board, _ = peek(create_env(strategist_game))
#     o, p, _, _ = peek(create_env(stumbler_game))

#     if debug:
#         print(">>> TRANING AN OPTIMAL STRATEGIST.")
#         print(">>> Train board {}".format(o, p))
#         print(">>> Test board {}".format(m, n))

#     # Log setup
#     if tensorboard:
#         writer = SummaryWriter(log_dir=path)

#     # Seeding...
#     np.random.seed(seed)

#     # Train params
#     strategic_default_value = 0.0
#     batch_size = 64

#     # ------------------------------------------------------------------------
#     # Build a Strategist, its memory, and its optimizer

#     # Create a model, of the right size.
#     # model = HotCold2(2, num_hidden1=num_hidden1)
#     model = HotCold3(2, num_hidden1=num_hidden1, num_hidden2=num_hidden2)

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     memory = ReplayMemory(10000)

#     # Run learning episodes. The 'stumbler' is just the opt
#     # cold board
#     for episode in range(num_episodes):
#         # The cold spots are '1' everythig else is '0'
#         strategic_value = create_cold_board(o, p)

#         # ...Into tuples
#         s_data = convert_ijv(strategic_value)
#         s_data = balance_ijv(s_data, strategic_default_value)

#         for d in s_data:
#             memory.push(*d)

#         loss = 0.0
#         if len(memory) > batch_size:
#             # Sample data....
#             coords = []
#             values = []
#             samples = memory.sample(batch_size)

#             for c, v in samples:
#                 coords.append(c)
#                 values.append(v)

#             coords = torch.tensor(
#                 np.vstack(coords), requires_grad=True, dtype=torch.float)
#             values = torch.tensor(
#                 values, requires_grad=False, dtype=torch.float)

#             # Making some preditions,
#             predicted_values = model(coords).squeeze()

#             # and find their loss.
#             loss = F.mse_loss(predicted_values, values)

#             # Walk down the hill of righteousness!
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if debug:
#                 print(">>> Coords: {}".format(coords))
#                 print(">>> Values: {}".format(values))
#                 print(">>> Predicted values: {}".format(values))
#                 print(">>> Loss {}".format(loss))

#         # Use the trained strategist to generate a bias_board,
#         bias_board = create_bias_board(m, n, model)

#         if tensorboard and (int(episode) % 50) == 0:
#             writer.add_scalar(os.path.join(path, 'error'), loss, episode)

#             plot_wythoff_board(
#                 strategic_value,
#                 vmin=0,
#                 vmax=1,
#                 path=path,
#                 name='strategy_board.png')
#             writer.add_image(
#                 'Training board',
#                 skimage.io.imread(os.path.join(path, 'strategy_board.png')))

#             plot_wythoff_board(
#                 bias_board, vmin=0, vmax=1, path=path, name='bias_board.png')
#             writer.add_image(
#                 'Testing board',
#                 skimage.io.imread(os.path.join(path, 'bias_board.png')))

#     # The end
#     if tensorboard:
#         writer.close()

#     return model

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# HELPER FNs


def add_bias_board(Qs, available, bias_board, influence):
    """Add bias to Qs."""

    assert len(Qs) == len(available), "Qs/available mismatch."

    if bias_board is None:
        return Qs
    if np.isclose(influence, 0.0):
        return Qs

    for i, (x, y) in enumerate(available):
        Qs[i] = Qs[i] + (influence * bias_board[x, y])

    return Qs


def estimate_strategic_value(m, n, hotcold):
    """Create a board to bias a stumblers moves."""

    strategic_value = np.zeros((m, n))
    with torch.no_grad():
        for i in range(m):
            for j in range(n):
                coord = torch.tensor([i, j], dtype=torch.float)
                strategic_value[i, j] = hotcold(coord)

    return strategic_value


def convert_ijv(data):
    """Convert a (m,n) matrix into a list of (i, j, value)"""

    m, n = data.shape
    converted = []
    for i in range(m):
        for j in range(n):
            converted.append([(i, j), data[i, j]])

    return converted


def balance_ijv(ijv_data, target_value):
    """Balance counts of target versus other values"""

    # Separate data based on target_value
    other = []
    null = []
    for c, v in ijv_data:
        if np.isclose(v, target_value):
            null.append([c, v])
        else:
            other.append([c, v])

    # Sizes
    N_null = len(null)
    N_other = len(other)

    # Sanity?
    if N_null == 0:
        return None
    if N_other == 0:
        return None

    np.random.shuffle(null)
    np.random.shuffle(other)

    # Finally, balance one or the other. Who is bigger?
    if N_null > N_other:
        null = null[:N_other]
    elif N_other > N_null:
        other = other[:N_null]
    else:
        # They already balanced. Return I
        return ijv_data

    return null + other


def expected_value(m, n, model, default_value=0.0):
    """Estimate the max value of each board position"""

    values = np.zeros((m, n))
    all_possible_moves = create_all_possible_moves(m, n)
    for i in range(m):
        for j in range(n):
            board = tuple(flatten_board(create_board(i, j, m, n)))
            try:
                v = model[board].max()
                values[i, j] = v
            except KeyError:
                values[i, j] = default_value

    return values


def estimate_cold(m, n, model, threshold=0.0, value=-1, default_value=0.0):
    """Estimate cold positions, enforcing symmetry on the diagonal"""
    values = expected_value(m, n, model, default_value=default_value)
    cold = np.zeros_like(values)

    # Cold
    mask = values < threshold
    cold[mask] = value
    cold[mask.transpose()] = value

    return cold


def estimate_hot(m, n, model, threshold=0.5, value=1, default_value=0.0):
    """Estimate hot positions"""
    values = expected_value(m, n, model, default_value=default_value)
    hot = np.zeros_like(values)

    mask = values > threshold
    hot[mask] = value

    return hot


def estimate_hot_cold(m,
                      n,
                      model,
                      hot_threshold=0.5,
                      cold_threshold=0.25,
                      cold_value=-1,
                      hot_value=1,
                      default_value=0.0):
    """Estimate hot and cold positions"""
    hot = estimate_hot(
        m,
        n,
        model,
        hot_threshold,
        value=hot_value,
        default_value=default_value)
    cold = estimate_cold(
        m,
        n,
        model,
        cold_threshold,
        value=cold_value,
        default_value=default_value)

    return hot + cold


def pad_board(m, n, board, value):
    """Given a board-shaped array, pad it to (m,n) with value."""

    padded = np.ones((m, n)) * value
    o, p = board.shape
    padded[0:o, 0:p] = board

    return padded


def estimate_alp_hot_cold(m, n, model, conf=0.05, default=0.5):
    """Estimate hot and cold positions"""

    values = expected_value(m, n, model)
    values = (values + 1.0) / 2.0

    hotcold = np.ones_like(values) * default

    # Cold
    mask = values < (conf * 1.3)
    hotcold[mask] = values[mask]
    hotcold[mask.transpose()] = values[mask]

    # Hot?
    # TODO? Skipped the random part....
    mask = values > (1 - conf)
    hotcold[mask] = values[mask]

    return hotcold


def create_bias_board(m, n, strategist_model, default=0.0):
    """"Sample all positions' value in a strategist model"""
    bias_board = torch.ones((m, n), dtype=torch.float) * default

    with torch.no_grad():
        for i in range(m):
            for j in range(n):
                coords = torch.tensor([i, j], dtype=torch.float)
                bias_board[i, j] = strategist_model(coords)

    return bias_board


def peek(env):
    """Peak at the env's board"""
    x, y, board, moves = env.reset()
    env.close()
    m, n = board.shape

    return m, n, board, moves


def flatten_board(board):
    m, n = board.shape
    return board.reshape(m * n)


def create_env(wythoff_name):
    env = gym.make('{}-v0'.format(wythoff_name))
    env = wrappers.Monitor(
        env, './tmp/{}-v0-1'.format(wythoff_name), force=True)

    return env


def plot_wythoff_board(board,
                       vmin=-1.5,
                       vmax=1.5,
                       plot=False,
                       path=None,
                       height=2,
                       width=3,
                       name='wythoff_board.png'):
    """Plot the board"""

    fig, ax = plt.subplots(figsize=(width, height))  # Sample figsize in inches
    ax = sns.heatmap(board, linewidths=3, vmin=vmin, vmax=vmax, ax=ax)

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.01)

    plt.close('all')


def evaluate_models(stumbler,
                    strategist,
                    stumbler_game,
                    strategist_game,
                    num_episodes=100,
                    mode='random',
                    debug=False):
    """Compare stumblers to strategists.
    
    Returns 
    -------
    wins : float
        the fraction of games won by the strategist.
    """
    # ------------------------------------------------------------------------
    # Init boards, etc
    # Stratgist
    env = create_env(strategist_game)
    m, n, board, _ = peek(env)
    hot_cold_table = create_bias_board(m, n, strategist)

    # Stumbler
    o, p, _, _ = peek(create_env(stumbler_game))

    # ------------------------------------------------------------------------
    # A stumbler and a strategist take turns playing a (m,n) game of wythoffs
    wins = 0.0
    for episode in range(num_episodes):
        # Re-init
        steps = 0

        # Start the game, and process the result
        x, y, board, available = env.reset()
        board = tuple(flatten_board(board))

        if debug:
            print("---------------------------------------")
            print(">>> NEW MODEL EVALUATION ({}).".format(episode))
            print(">>> Initial position ({}, {})".format(x, y))
            print(">>> Initial moves {}".format(available))

        done = False
        while not done:
            # ----------------------------------------------------------------
            # STUMBLER
            if (x < o) and (y < p):
                s_board = tuple(flatten_board(create_board(x, y, o, p)))
                s_available = create_moves(x, y)
                try:
                    values = stumbler[s_board]
                    move_i = epsilon_greedy(values, epsilon=0.0, mode='numpy')
                    move = s_available[move_i]
                except KeyError:
                    move_i = np.random.randint(0, len(available))
                    move = available[move_i]
            else:
                move_i = np.random.randint(0, len(available))
                move = available[move_i]

            (x, y, board, available), reward, done, _ = env.step(move)
            board = tuple(flatten_board(board))
            if debug:
                print(">>> STUMBLER move {}".format(move))

            if done:
                break

            # ----------------------------------------------------------------
            # STRATEGIST
            hot_cold_move_values = [hot_cold_table[x, y] for x, y in available]

            move_i = epsilon_greedy(
                np.asarray(hot_cold_move_values), epsilon=0.0, mode='numpy')
            move = available[move_i]

            if debug:
                print(">>> STRATEGIST move {}".format(move))

            # Make a move
            (x, y, board, available), reward, done, _ = env.step(move)
            board = tuple(flatten_board(board))
            if done:
                wins += 1.0

        if debug:
            print("Wins {}".format(wins / episode + 1))

    return wins / num_episodes
