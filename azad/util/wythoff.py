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
from azad.policy import epsilon_greedy
from azad.policy import greedy
from azad.local_gym.wythoff import create_moves
from azad.local_gym.wythoff import create_all_possible_moves
from azad.local_gym.wythoff import locate_moves
from azad.local_gym.wythoff import create_cold_board
from azad.local_gym.wythoff import create_board


def plot_wythoff_board(board,
                       vmin=-1.5,
                       vmax=1.5,
                       plot=False,
                       path=None,
                       name='wythoff_board.png'):
    """Plot the board"""

    fig, ax = plt.subplots()  # Sample figsize in inches
    ax = sns.heatmap(board, linewidths=3, vmin=vmin, vmax=vmax, ax=ax)

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.01)

    plt.close('all')


def plot_cold_board(m, n, plot=False, path=None, name='cold_board.png'):
    cold = create_cold_board(m, n)

    fig, ax = plt.subplots()  # Sample figsize in inches
    ax = sns.heatmap(cold, linewidths=3, ax=ax)

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.1)

    plt.close('all')


def expected_value(m, n, model):
    """Estimate the expected value of each board position"""

    # Build the matrix a values for each (i, j) board
    values = np.zeros((m, n))
    all_possible_moves = create_all_possible_moves(m, n)

    with torch.no_grad():
        for i in range(m):
            for j in range(n):
                board = flatten_board(create_board(i, j, m, n))
                if i == 0 and j == 0:
                    moves = [(0, 0)]
                else:
                    moves = create_moves(i, j)

                index = locate_moves(moves, all_possible_moves)
                values[i, j] = np.max(model(board).detach().numpy()[index])

    return values


def plot_wythoff_expected_values(m,
                                 n,
                                 model,
                                 plot=False,
                                 path=None,
                                 vmin=-2,
                                 vmax=2,
                                 name='wythoff_expected_values.png'):
    """Plot EVs"""
    values = expected_value(m, n, model)

    # !
    fig, ax = plt.subplots()  # Sample figsize in inches
    ax = sns.heatmap(values, linewidths=3, vmin=vmin, vmax=vmax, ax=ax)
    # ax = sns.heatmap(values, linewidths=3, ax=ax)

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        plt.pause(0.01)

    plt.close()


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


def balance_ijv(ijv_data, null_value):
    """Balance counts of null versus other values"""

    # Separate data based on null_value
    other = []
    null = []
    for c, v in ijv_data:
        if np.isclose(v, null_value):
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


def estimate_cold(m, n, model, threshold=0.0):
    """Estimate cold positions, enforcing symmetry on the diagonal"""
    values = expected_value(m, n, model)
    cold = np.zeros_like(values)

    # Cold
    mask = values < threshold
    cold[mask] = -1.0
    cold[mask.transpose()] = -1.0

    return cold


def estimate_hot(m, n, model, threshold=0.5):
    """Estimate hot positions"""
    values = expected_value(m, n, model)
    hot = np.zeros_like(values)

    mask = values > threshold
    hot[mask] = 1.0

    return hot


def estimate_hot_cold(m, n, model, hot_threshold=0.5, cold_threshold=0.25):
    """Estimate hot and cold positions"""
    hot = estimate_hot(m, n, model, hot_threshold)
    cold = estimate_cold(m, n, model, cold_threshold)

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


def create_q_table(m, n, a, model, default=0.0):
    """Estimate the expected value of each board position"""
    # Build the matrix a values for each (i, j) board
    qs = torch.zeros((m, n, a), dtype=torch.float)

    with torch.no_grad():
        for i in range(m):
            for j in range(n):
                board = create_board(i, j, m, n)
                board = torch.tensor(board.reshape(m * n), dtype=torch.float)
                qs[i, j, :] = model(board)

    return qs


def evaluate_models(stumbler,
                    strategist,
                    stumbler_env,
                    strategist_env,
                    num_eval=100,
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
    m, n, board, _ = peek(strategist_env)
    all_strategist_moves = create_all_possible_moves(m, n)
    all_strategist_moves_index = np.arange(0, len(all_strategist_moves))
    hot_cold_table = create_bias_board(m, n, strategist)

    # Stumbler
    o, p, board, _ = peek(stumbler_env)
    all_stumbler_moves = create_all_possible_moves(o, p)
    all_stumbler_moves_index = np.arange(0, len(all_stumbler_moves))
    q_table = create_q_table(o, p, len(all_stumbler_moves), stumbler)

    # ------------------------------------------------------------------------
    # A stumbler and a strategist take turns playing a (m,n) game of wythoffs
    wins = 0.0
    for trial in range(num_eval):
        # --------------------------------------------------------------------
        # Re-init
        steps = 0

        # Start the game, and process the result
        x, y, board, moves = strategist_env.reset()
        board = flatten_board(board)

        if debug:
            print("---------------------------------------")
            print(">>> NEW MODEL EVALUATION ({}).".format(trial))
            print(">>> Initial position ({}, {})".format(x, y))
            print(">>> Initial moves {}".format(moves))

        done = False
        while not done:
            # ------------------------------------------------------------
            # STUMBLER

            # Choose a move
            if (x < o) and (y < p):
                moves_index = locate_moves(moves, all_stumbler_moves)
                move_i = greedy(q_table[x, y, :], index=moves_index)
                move = all_stumbler_moves[move_i]
            else:
                moves_index = locate_moves(moves, all_strategist_moves)
                move_i = np.random.choice(moves_index)
                move = all_strategist_moves[move_i]

            if debug:
                print(">>> STUMBLER move {}".format(move))

            # Make a move
            (x, y, board, moves), reward, done, _ = strategist_env.step(move)
            board = flatten_board(board)
            moves_index = locate_moves(moves, all_strategist_moves)

            # Die early if the stumbler wins
            if done:
                if debug: print("*** STUMBLER WIN ***")
                break

            # ------------------------------------------------------------
            # STRATEGIST

            # Choose a move
            move_i = greedy(flatten_board(hot_cold_table), index=moves_index)
            move = all_strategist_moves[move_i]

            if debug:
                print(">>> STRATEGIST move {}".format(move))

            # Make a move
            (x, y, board, moves), reward, done, _ = strategist_env.step(move)
            board = flatten_board(board)
            moves_index = locate_moves(moves, all_strategist_moves)

            if done:
                wins += 1.0
                if debug: print("*** STRATEGIST WIN ***")

    return wins / num_eval


def peek(env):
    """Peak at the env's board"""
    x, y, board, moves = env.reset()
    env.close()
    m, n = board.shape

    return m, n, board, moves


def flatten_board(board):
    m, n = board.shape
    return torch.tensor(board.reshape(m * n), dtype=torch.float)


def random_action(possible_actions):
    idx = np.random.randint(0, len(possible_actions))
    return idx, possible_actions[idx]


def create_env(wythoff_name):
    env = gym.make('{}-v0'.format(wythoff_name))
    env = wrappers.Monitor(
        env, './tmp/{}-v0-1'.format(wythoff_name), force=True)

    return env


def _np_greedy(x, index=None):
    """Pick the biggest"""
    # Filter x using index, but first ensure we can
    # map the action back to x' orignal 'space'
    if index is not None:
        x = x[index]

    action = np.argmax(x)

    # Map back to x's original space
    if index is not None:
        action = index[action]

    return action


def _np_epsilon_greedy(x, epsilon, index=None):
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


def _np_softmax(x, beta=0.98, index=None):
    """
    Pick an action with softmax
    """

    # Filter x using index, but first ensure we can
    # map the action back to x' orignal 'space'
    if index is not None:
        x = x[index]

    # multiply x against the beta parameter,
    x = x * float(beta)

    # subtract the max for numerical stability
    x = x - np.max(x)

    # exponentiate x
    x = np.exp(x)

    # take the sum along the specified axis
    ax_sum = np.sum(x)

    # finally: divide elementwise
    p = x / ax_sum

    # sample action using p
    actions = np.arange(len(x), dtype=np.int)
    action = np.random.choice(actions, p=p)

    # Map back to x's original space
    if index is not None:
        action = index[action]

    return action


def _np_expected_value(m, n, model):
    """Estimate the expected value of each board position"""
    # Build the matrix a values for each (i, j) board
    values = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            board = tuple(flatten_board(create_board(i, j, m, n)).numpy())
            try:
                values[i, j] = model[board].mean()
            except KeyError:
                values[i, j] = 0.0

    return values


def _np_min_value(m, n, model):
    """Estimate the min value of each board position"""
    # Build the matrix a values for each (i, j) board
    values = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            board = tuple(flatten_board(create_board(i, j, m, n)).numpy())
            try:
                values[i, j] = model[board].min()
            except KeyError:
                values[i, j] = 0.0

    return values


def _np_max_value(m, n, model):
    """Estimate the max value of each board position"""
    # Build the matrix a values for each (i, j) board
    values = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            board = tuple(flatten_board(create_board(i, j, m, n)).numpy())
            try:
                values[i, j] = model[board].max()
            except KeyError:
                values[i, j] = 0.0

    return values


def _np_plot_wythoff_expected_values(m,
                                     n,
                                     model,
                                     plot=False,
                                     path=None,
                                     vmin=-2,
                                     vmax=2,
                                     name='wythoff_expected_values.png'):
    """Plot EVs"""
    values = _np_expected_value(m, n, model)

    # !
    fig, ax = plt.subplots()  # Sample figsize in inches
    # ax = sns.heatmap(values, linewidths=3, vmin=vmin, vmax=vmax, ax=ax)
    ax = sns.heatmap(values, linewidths=3, ax=ax)

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.01)

    plt.close()


def _np_plot_wythoff_min_values(m,
                                n,
                                model,
                                plot=False,
                                path=None,
                                vmin=-2,
                                vmax=2,
                                name='wythoff_min_values.png'):
    """Plot min Vs"""
    values = _np_min_value(m, n, model)

    # !
    fig, ax = plt.subplots()  # Sample figsize in inches
    # ax = sns.heatmap(values, linewidths=3, vmin=vmin, vmax=vmax, ax=ax)
    ax = sns.heatmap(values, linewidths=3, ax=ax)

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.01)

    plt.close()


def _np_plot_wythoff_max_values(m,
                                n,
                                model,
                                plot=False,
                                path=None,
                                vmin=-2,
                                vmax=2,
                                name='wythoff_max_values.png'):
    """Plot  max Vs"""
    values = _np_max_value(m, n, model)

    # !
    fig, ax = plt.subplots()  # Sample figsize in inches
    # ax = sns.heatmap(values, linewidths=3, vmin=vmin, vmax=vmax, ax=ax)
    ax = sns.heatmap(values, linewidths=3, ax=ax)

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.01)

    plt.close()