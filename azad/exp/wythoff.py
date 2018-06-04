import os, csv
import sys

import errno

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import golden

import skimage
from skimage import data, io

import gym
from gym import wrappers
import azad.local_gym

from azad.models import OneLinQN
from azad.models import HotCold
from azad.policy import epsilon_greedy
from azad.policy import greedy
from azad.util import ReplayMemory

# # ---------------------------------------------------------------
# # Handle dtypes for the device
# USE_CUDA = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
# Tensor = FloatTensor

# # ---------------------------------------------------------------


def plot_wythoff_board(board, plot=False, path=None, name='wythoff_board.png'):
    """Plot the board"""

    # !
    fig = plt.figure()
    plt.matshow(board)
    plt.colorbar()

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.01)

    plt.close()


def plot_cold_board(m, n, plot=False, path=None, name='cold_board.png'):
    cold = create_cold_board(m, n)

    # !
    fig = plt.figure()
    plt.matshow(cold)
    plt.colorbar()

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.1)

    plt.close()


def plot_q_action_values(m,
                         n,
                         a,
                         model,
                         plot=False,
                         possible_actions=None,
                         path=None,
                         name='wythoff_q_action_values.png'):
    """Plot Q values for each possible action"""

    qs = estimate_q_values(m, n, a, model)

    # Init plot
    fig, axarr = plt.subplots(1, a, sharey=True, figsize=(17, 4))

    for k in range(a):
        q_a = qs[:, :, k]
        im = axarr[k].matshow(q_a)
        fig.colorbar(im, ax=axarr[k])

        if possible_actions is not None:
            axarr[k].set_title(possible_actions[k])

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.01)

    plt.close('all')


def plot_wythoff_expected_values(m,
                                 n,
                                 model,
                                 plot=False,
                                 path=None,
                                 name='wythoff_expected_values.png'):
    """Plot EVs"""
    values = expected_value(m, n, model)

    # !
    fig = plt.figure()
    plt.matshow(values)
    plt.colorbar()

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.01)

    plt.close()


def create_cold_board(m, n):
    cold_board = np.zeros((m, n))
    for k in range(m - 1):
        mk = int(k * golden)
        nk = int(k * golden**2)
        if (nk < m) and (mk < n):
            cold_board[mk, nk] = -1
            cold_board[nk, mk] = -1

    return cold_board


def create_board(i, j, m, n):
    board = np.zeros((m, n))
    board[i, j] = 1.0

    return board


def create_bias_board(m, n, hotcold):
    """Create a board to bias a stumblers moves."""

    bias_board = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            coord = torch.tensor([i, j], dtype=torch.float)
            bias_board[i, j] = hotcold(coord)

    return bias_board


def convert_ijv(data):
    """Convert a (m,n) matrix into a list of (i, j, value)"""

    m, n = data.shape
    converted = []
    for i in range(m):
        for j in range(n):
            converted.append([(i, j), data[i, j]])

    return converted


def estimate_q_values(m, n, a, model):
    """Estimate the expected value of each board position"""
    # Build the matrix a values for each (i, j) board
    qs = np.zeros((m, n, a))
    for i in range(m):
        for j in range(n):
            board = create_board(i, j, m, n)
            board = torch.tensor(board.reshape(m * n), dtype=torch.float)
            qs[i, j, :] = model(board).detach().numpy()

    return qs


def expected_value(m, n, model):
    """Estimate the expected value of each board position"""
    # Build the matrix a values for each (i, j) board
    values = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            board = create_board(i, j, m, n)
            board = torch.tensor(board.reshape(m * n), dtype=torch.float)
            values[i, j] = np.max(model(board).detach().numpy())

    return values


def estimate_cold(m, n, model, threshold=0.25):
    """Estimate cold positions, enforcing symmetry on the diagonal"""
    values = expected_value(m, n, model)
    cold = np.zeros_like(values)

    # Cold
    mask = values < threshold
    cold[mask] = -1.0
    cold[mask.transpose()] = -1.0

    return cold


def estimate_hot(m, n, model, threshold=0.75):
    """Estimate hot positions"""
    values = expected_value(m, n, model)
    hot = np.zeros_like(values)

    mask = values > threshold
    hot[mask] = 1.0

    return hot


def estimate_hot_cold(m, n, model, hot_threshold=0.75, cold_threshold=0.25):
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


def expected_alp_value(m, n, model):
    """Estimate the expected value using the 'Muyesser' method
    
    As described in:
    
    Muyesser, N.A., Dunovan, K. & Verstynen, T., 2018. Learning model-based 
    strategies in simple environments with hierarchical q-networks. , pp.1–29. A
    vailable at: http://arxiv.org/abs/1801.06689.
    """

    values = expected_value(m, n, model)
    values = (values + 1.0) / 2.0

    return values


def estimate_alp_cold(m, n, model, threshold=0.25, default_value=0.5):
    """Estimate cold positions, enforcing symmetry on the diagonal"""
    values = expected_alp_value(m, n, model)
    cold = np.ones_like(values) * default_value

    # Cold
    mask = values < threshold
    cold[mask] = values[mask]
    cold[mask.transpose()] = values[mask]

    return cold


def estimate_alp_hot(m, n, model, threshold=0.75, default_value=0.5):
    """Estimate hot positions"""

    values = expected_alp_value(m, n, model)
    hot = np.ones_like(values) * default_value

    mask = values > threshold
    hot[mask] = values[mask]

    return hot


def estimate_alp_hot_cold(m, n, hot_threshold=0.75, cold_threshold=0.25):
    """Estimate hot and cold positions"""
    hot = estimate_alp_hot(m, n, hot_threshold)
    cold = estimate_alp_cold(m, n, cold_threshold)

    return hot, cold


def evauluate_models(stumbler,
                     strategist,
                     stumbler_env,
                     strategist_env,
                     num_eval=100,
                     possible_actions=[(-1, 0), (0, -1), (-1, -1)]):
    """Compare stumblers to strategists.
    
    Returns 
    -------
    wins : float
        the fraction of games won by the strategist.
    """
    # -------------------------------------------
    # Init boards, etc

    # Stratgist
    x, y, board = strategist_env.reset()
    strategist_env.close()
    m, n = board.shape

    strategy_values = create_bias_board(m, n, strategist)

    # Stumbler
    _, _, little_board = stumbler_env.reset()
    o, p = little_board.shape
    stumbler_env.close()

    q_values = estimate_q_values(o, p, len(possible_actions), stumbler)

    # -------------------------------------------
    # a stumbler and a strategist take turns
    # playing a game....
    wins = 0.0
    for _ in range(num_eval):
        # (re)init
        x, y, board = strategist_env.reset()
        board = torch.tensor(board.reshape(m * n))

        while True:
            # -------------------------------------------
            # The stumbler goes first (the polite and 
            # coservative thing to do).
            # (Adj for board size differences)
            if (x < o) and (y < p):
                Qs = q_values[x, y, :]
                Qs = torch.tensor(Qs)

                action_index = greedy(Qs)
            else:
                action_index = np.random.randint(0, len(possible_actions))

            action = possible_actions[int(action_index)]

            (x, y, _), reward, done, _ = strategist_env.step(action)
            if done:
                break

            # Now the strategist
            Vs = []
            for a in possible_actions:
                Vs.append(strategy_values[x + a[0], y + a[1]])

            action_index = greedy(torch.tensor(Vs))
            action = possible_actions[int(action_index)]

            (x, y, _), reward, done, _ = strategist_env.step(action)
            if done:
                wins += 1  # We care when the strategist wins
                break

    return float(wins)


def peak(name):
    """Peak at the env's board"""
    env = gym.make('{}-v0'.format(name))

    x, y, board = env.reset()
    env.close()
    m, n = board.shape

    return m, n, board


def wythoff_stumbler(path,
                     num_trials=10,
                     epsilon=0.1,
                     gamma=0.8,
                     learning_rate=0.1,
                     wythoff_name='Wythoff3x3',
                     model=None,
                     bias_board=None,
                     log=False,
                     seed=None):
    """Train a Q-agent to play Wythoff's game, using SGD."""

    # -------------------------------------------
    # Create path
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # -------------------------------------------
    # Log setup    
    if log:
        # Create a csv for archiving select data
        f = open(os.path.join(path, "data.csv"), "w")
        csv_writer = csv.writer(f)
        csv_writer.writerow(["trial", "steps", "action_index", "Q", "x", "y"])

        # Tensorboard log
        writer = SummaryWriter(log_dir=path)

    # -------------------------------------------
    # The world is a pebble on a board
    env = gym.make('{}-v0'.format(wythoff_name))
    env = wrappers.Monitor(
        env, './tmp/{}-v0-1'.format(wythoff_name), force=True)

    # -------------------------------------------
    # Seeding...
    env.seed(seed)
    np.random.seed(seed)

    # -------------------------------------------
    # Valid moves (in this simplified instantiation)
    possible_actions = [(-1, 0), (0, -1), (-1, -1)]

    # -------------------------------------------
    # Build a Q agent, its memory, and its optimizer

    # How big is the board?
    m, n, board = peak(wythoff_name)

    # Create a model of the right size?
    if model is None:
        model = OneLinQN(m * n, len(possible_actions))

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # -------------------------------------------
    # Run some trials
    for trial in range(num_trials):
        x, y, board = env.reset()
        board = torch.tensor(board.reshape(m * n), dtype=torch.float)

        steps = 0
        while True:
            # -------------------------------------------
            # TODO
            # env.render()

            # -------------------------------------------
            # Look at the world and approximate its value.
            Qs = model(board).squeeze()

            # Use the bias_board to bias Qs
            # based on all possible next moves.
            if bias_board is not None:
                Qs_bias = np.zeros_like(Qs.detach())
                for i, a in enumerate(possible_actions):
                    Qs_bias[i] = bias_board[x + a[0], y + a[1]]

                Qs += torch.tensor(Qs_bias, dtype=torch.float)

            # Make a decision.
            action_index = epsilon_greedy(Qs, epsilon)
            action = possible_actions[int(action_index)]

            Q = Qs[int(action_index)]

            (x, y, next_board), reward, done, _ = env.step(action)
            next_board = torch.tensor(
                next_board.reshape(m * n), dtype=torch.float)

            # Update move counter
            steps += 1

            # ---
            # Learn w/ SGD
            max_Q = model(next_board).detach().max()
            next_Q = reward + (gamma * max_Q)
            loss = F.smooth_l1_loss(Q, next_Q)

            # Walk down the hill of righteousness!
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # retain is needed for opp. WHY?
            optimizer.step()

            # -------------------------------------------
            # Log results
            if log:
                csv_writer.writerow(
                    [trial, steps, int(action_index), float(Q), x, y])

                writer.add_scalar(os.path.join(path, 'Q'), Q, trial)
                writer.add_scalar(os.path.join(path, 'reward'), reward, trial)

            # -------------------------------------------
            # If the game is over, stop
            if done:
                break

            # -------------------------------------------
            # Otherwise the opponent moves
            action_index = np.random.randint(0, len(possible_actions))
            action = possible_actions[action_index]

            # Q = Qs[int(action_index)].squeeze()

            (x, y, next_board), reward, done, info = env.step(action)
            next_board = torch.tensor(
                next_board.reshape(m * n), dtype=torch.float)

            # # Flip signs so opp victories are punishments
            # if reward > 0:
            #     reward *= -1

            # # -
            # if log:
            #     writer.add_scalar(os.path.join(path, 'opp_Q'), Q, trial)
            #     writer.add_scalar(
            #         os.path.join(path, 'opp_reward'), reward, trial)

            # ---
            # Learn from your opponent
            # max_Q = model(next_board).detach().max()
            # next_Q = reward + (gamma * max_Q)
            # loss = F.smooth_l1_loss(Q, next_Q)

            # # Walk down the hill of righteousness!
            # loss.backward()
            # optimizer.step()

            # Plot?
            if (trial % 10) == 0 and log:
                plot_cold_board(m, n, path=path, name='cold_board.png')
                writer.add_image(
                    'cold_positions',
                    skimage.io.imread(os.path.join(path, 'cold_board.png')))

                plot_wythoff_expected_values(
                    m, n, model, path=path, name='wythoff_expected_values.png')
                writer.add_image(
                    'expected_value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_expected_values.png')))

                plot_q_action_values(
                    m,
                    n,
                    len(possible_actions),
                    model,
                    possible_actions=possible_actions,
                    path=path,
                    name='wythoff_q_action_values.png')
                writer.add_image(
                    'q_action_values',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_q_action_values.png')))

            if done:
                break

            # Shuffle state notation
            board = next_board

        # save min loss for this trial
        if log:
            writer.add_scalar(os.path.join(path, 'error'), loss.data[0], trial)

    # Cleanup and end
    if log:
        writer.close()
        f.close()

    return model, env


def wythoff_strategist(path,
                       num_trials=1000,
                       num_stumbles=100,
                       epsilon=0.1,
                       gamma=0.8,
                       delta=0.1,
                       learning_rate=0.1,
                       strategy_name='estimate_cold',
                       strategy_kwargs=None,
                       strategy_default_value=0,
                       wythoff_name_stumbler='Wythoff15x15',
                       wythoff_name_strategist='Wythoff50x50',
                       log=False,
                       seed=None):

    # -------------------------------------------
    # Setup
    # -------------------------------------------
    env = gym.make('{}-v0'.format(wythoff_name_strategist))
    env = wrappers.Monitor(
        env, './tmp/{}-v0-1'.format(wythoff_name_strategist), force=True)

    possible_actions = [(-1, 0), (0, -1), (-1, -1)]

    batch_size = 128
    num_strategist_iter = 1000

    # -------------------------------------------
    # Log setup
    if log:
        # Tensorboard log
        writer = SummaryWriter(log_dir=path)

    # -------------------------------------------
    # Seeding...
    env.seed(seed)
    np.random.seed(seed)

    # -------------------------------------------
    # Build a Strategist, its memory, and its optimizer

    # How big are the boards?
    m, n, board = peak(wythoff_name_strategist)
    o, p, _ = peak(wythoff_name_stumbler)

    # Create a model, of the right size.
    model = HotCold(2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    memory = ReplayMemory(10000)

    # Setup strategy
    this_module = sys.modules[__name__]
    strategy = getattr(this_module, strategy_name)
    if strategy_kwargs is not None:
        strategy = lambda m, n, model: strategy(m, n, model, **strategy_kwargs)

    # -------------------------------------------
    # !
    wins = []
    stumbler_model = None
    bias_board = None
    for trial in range(num_trials):

        stumbler_model, stumbler_env = wythoff_stumbler(
            path,
            num_trials=num_stumbles,
            epsilon=epsilon,
            gamma=gamma,
            wythoff_name=wythoff_name_stumbler,
            model=None,
            bias_board=None,
            learning_rate=learning_rate)

        # Extract strategic data from the stumber, 
        # project it and remember that
        strategy_board = strategy(o, p, stumbler_model)
        strategy_board = pad_board(m, n, strategy_board,
                                   strategy_default_value)

        # ...Into tuples
        s_data = convert_ijv(strategy_board)
        for d in s_data:
            memory.push(*d)

        # Train a strategist, by first sampling its memory
        for _ in range(num_strategist_iter):
            coords = []
            values = []
            samples = memory.sample(batch_size)

            for c, v in samples:
                coords.append(c)
                values.append(v)

            coords = torch.tensor(
                np.vstack(coords), requires_grad=True, dtype=torch.float)
            values = torch.tensor(
                values, requires_grad=True, dtype=torch.float)

            # Making some preditions,
            predicted_values = model(coords).detach().squeeze()

            # and find their loss.
            loss = F.smooth_l1_loss(values, predicted_values)

            # Walk down the hill of righteousness!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compare strategist and stumbler. Count strategist wins.
        win = evauluate_models(stumbler_model, model, stumbler_env, env)

        # Use the trained strategist to generate a bias_board,
        bias_board = create_bias_board(m, n, model)
        # bias_board[np.abs(bias_board) < delta] = 0.0

        if log:
            writer.add_scalar(
                os.path.join(path, 'stategist_error'), loss.data[0], trial)

            writer.add_scalar(os.path.join(path, 'stategist_wins'), win, trial)

            plot_wythoff_expected_values(o, p, stumbler_model, path=path)
            writer.add_image(
                'stumbler_expected_value',
                skimage.io.imread(
                    os.path.join(path, 'wythoff_expected_values.png')))

            plot_wythoff_board(
                strategy_board, path=path, name='strategy_board.png')
            writer.add_image(
                'stumbler_to_strategy_board',
                skimage.io.imread(os.path.join(path, 'strategy_board.png')))

            plot_wythoff_board(bias_board, path=path, name='bias_board.png')
            writer.add_image(
                'strategy_board',
                skimage.io.imread(os.path.join(path, 'bias_board.png')))

    return model, stumbler_model, env, stumbler_env, wins
