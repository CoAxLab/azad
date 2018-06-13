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

from azad.models import LinQN1
from azad.models import HotCold1
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


def plot_wythoff_board(board,
                       vmin=-1.5,
                       vmax=1.5,
                       plot=False,
                       path=None,
                       name='wythoff_board.png'):
    """Plot the board"""

    # !
    fig = plt.figure()
    plt.matshow(board, vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.01)

    plt.close('all')


def plot_cold_board(m, n, plot=False, path=None, name='cold_board.png'):
    cold = create_cold_board(m, n)

    # !
    fig = plt.figure()
    plt.matshow(cold, vmin=-1.5, vmax=1.5)
    plt.colorbar()

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, name))

    if plot:
        # plt.show()
        plt.pause(0.1)

    plt.close('all')


def plot_q_action_values(m,
                         n,
                         a,
                         model,
                         plot=False,
                         possible_actions=None,
                         path=None,
                         v_size=2,
                         name='wythoff_q_action_values.png'):
    """Plot Q values for each possible action"""

    qs = estimate_q_values(m, n, a, model)

    # Init plot
    fig, axarr = plt.subplots(1, a, sharey=True, figsize=(17, 4))

    for k in range(a):
        q_a = qs[:, :, k]
        im = axarr[k].matshow(q_a, vmin=-v_size, vmax=v_size)

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


def plot_random_rewards(m,
                        n,
                        a,
                        rewards,
                        plot=False,
                        possible_actions=None,
                        path=None,
                        name='wythoff_rewards.png'):
    """Plot sum rewards for each possible action"""

    # Init plot
    fig, axarr = plt.subplots(1, a, sharey=True, figsize=(17, 4))

    for k in range(a):
        r = rewards[:, :, k]
        im = axarr[k].matshow(r)
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
                                 v_size=2,
                                 name='wythoff_expected_values.png'):
    """Plot EVs"""
    values = expected_value(m, n, model)

    # !
    fig = plt.figure()
    plt.matshow(values, vmin=-v_size, vmax=v_size)
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
            cold_board[mk, nk] = 1
            cold_board[nk, mk] = 1

    return cold_board


def create_board(i, j, m, n):
    board = np.zeros((m, n))
    board[i, j] = 1.0

    return board


def estimate_strategic_value(m, n, hotcold):
    """Create a board to bias a stumblers moves."""

    strategic_value = np.zeros((m, n))
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
        raise ValueError("No null data to balance against.")

    # Nothing to do if there is no other data,
    # so just return I
    if N_other == 0:
        return ijv_data

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


def evaluate_models(stumbler,
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

    strategy_values = estimate_strategic_value(m, n, strategist)

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


def wythoff_random(path,
                   num_trials=10,
                   wythoff_name='Wythoff3x3',
                   log=False,
                   seed=None):
    """Play a game of random (valid) moves."""

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
    m, n, board = peak(wythoff_name)

    # -------------------------------------------
    # Seeding...
    env.seed(seed)
    np.random.seed(seed)

    # -------------------------------------------
    # Valid moves (in this simplified instantiation)
    possible_actions = [(-1, 0), (0, -1), (-1, -1)]
    a = len(possible_actions)

    # Create a place to sum rewards
    rewards = np.zeros((m, n, a))

    # Run the random trials....
    for trial in range(num_trials):
        x, y, board = env.reset()

        steps = 0
        while True:
            # Make a decision,
            action_index = np.random.randint(0, a)
            action = possible_actions[int(action_index)]

            # and a move.
            (x, y, next_board), reward, done, _ = env.step(action)
            rewards[x, y, action_index] += reward

            # Update move counter
            steps += 1

            if log:
                writer.add_scalar(os.path.join(path, 'steps'), steps, trial)
                writer.add_scalar(os.path.join(path, 'reward'), reward, trial)
                writer.add_scalar(
                    os.path.join(path, 'action_index'), action_index, trial)

            if done:
                break

        if (trial % 10) == 0 and log:
            # Plot?
            plot_random_rewards(
                m,
                n,
                len(possible_actions),
                rewards,
                possible_actions=possible_actions,
                path=path,
                name='wythoff_rewards.png')

            writer.add_image(
                'wythoff_rewards',
                skimage.io.imread(os.path.join(path, 'wythoff_rewards.png')))

    return rewards, env


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
        model = LinQN1(m * n, len(possible_actions))

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

            if log:
                writer.add_scalar(os.path.join(path, 'Q'), Q, trial)
                writer.add_scalar(os.path.join(path, 'reward'), reward, trial)
                writer.add_scalar(os.path.join(path, 'steps'), steps, trial)

            # -------------------------------------------
            # If the game is over, stop
            # Otherwise the opponent moves
            if done:
                break

            action_index = np.random.randint(0, len(possible_actions))
            action = possible_actions[action_index]

            Q = Qs[int(action_index)].squeeze()

            (x, y, next_board), reward, done, info = env.step(action)
            next_board = torch.tensor(
                next_board.reshape(m * n), dtype=torch.float)

            # Learn from your opponent when they win
            if done:
                # Opp victories are punishments
                reward *= -1

                max_Q = model(next_board).detach().max()
                next_Q = reward + (gamma * max_Q)
                loss = F.smooth_l1_loss(Q, next_Q)

                loss.backward()
                optimizer.step()

            if log:
                writer.add_scalar(os.path.join(path, 'opp_Q'), Q, trial)
                writer.add_scalar(
                    os.path.join(path, 'opp_reward'), reward, trial)

            if (trial % 10) == 0 and log:
                # Opt play
                plot_cold_board(m, n, path=path, name='cold_board.png')
                writer.add_image(
                    'cold_positions',
                    skimage.io.imread(os.path.join(path, 'cold_board.png')))

                # EV
                plot_wythoff_expected_values(
                    m, n, model, path=path, name='wythoff_expected_values.png')
                writer.add_image(
                    'expected_value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_expected_values.png')))

                # Q(a)
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

            # Stop or update?
            if done:
                break

            board = next_board

        if log:
            writer.add_scalar(os.path.join(path, 'error'), loss.data[0], trial)

    # The end
    if log:
        writer.close()

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
                       strategic_default_value=0.5,
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

    batch_size = 64
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
    model = HotCold1(2)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
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
        strategic_value = strategy(o, p, stumbler_model)
        strategic_value = pad_board(m, n, strategic_value,
                                    strategic_default_value)

        # ...Into tuples
        s_data = convert_ijv(strategic_value)
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

            # Scale coords
            coords = coords / m

            # Making some preditions,
            predicted_values = model(coords).detach().squeeze()

            # print(values[:4], predicted_values[:4])

            # and find their loss.
            # loss = F.smooth_l1_loss(values, predicted_values)
            # loss = F.mse_loss(values, predicted_values)
            loss = F.mse_loss(
                torch.clamp(values, 0, 1), torch.clamp(predicted_values, 0, 1))
            # loss = F.binary_cross_entropy(
            #     torch.clamp(values, 0, 1), torch.clamp(predicted_values, 0, 1))

            # Walk down the hill of righteousness!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compare strategist and stumbler. Count strategist wins.
        win = evaluate_models(stumbler_model, model, stumbler_env, env)

        # Use the trained strategist to generate a bias_board,
        bias_board = estimate_strategic_value(m, n, model)
        # bias_board[np.abs(bias_board) < delta] = 0.0

        if log:
            writer.add_scalar(
                os.path.join(path, 'stategist_error'), loss.data[0], trial)

            writer.add_scalar(os.path.join(path, 'stategist_wins'), win, trial)

            plot_wythoff_expected_values(
                o, p, stumbler_model, path=path, v_size=3)
            writer.add_image(
                'stumbler_expected_value',
                skimage.io.imread(
                    os.path.join(path, 'wythoff_expected_values.png')))

            plot_wythoff_board(
                strategic_value, path=path, name='strategy_board.png')
            writer.add_image(
                'stumbler_to_strategy_board',
                skimage.io.imread(os.path.join(path, 'strategy_board.png')))

            plot_wythoff_board(bias_board, path=path, name='bias_board.png')
            writer.add_image(
                'strategy_board',
                skimage.io.imread(os.path.join(path, 'bias_board.png')))

            plot_q_action_values(
                o,
                p,
                len(possible_actions),
                stumbler_model,
                v_size=3,
                possible_actions=possible_actions,
                path=path,
                name='wythoff_q_action_values.png')

            writer.add_image(
                'stumbler_q_action_values',
                skimage.io.imread(
                    os.path.join(path, 'wythoff_q_action_values.png')))

    return model, stumbler_model, env, stumbler_env, wins
