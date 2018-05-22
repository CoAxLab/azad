import os, csv
import errno

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import data, io

import gym
from gym import wrappers
import azad.local_gym

from azad.models import OneLinQN
from azad.policy import epsilon_greedy

# ---------------------------------------------------------------
# Handle dtypes for the device
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor

# ---------------------------------------------------------------


def plot_wythoff_board(board, plot=False, path=None):
    """Plot the board"""

    # !
    fig = plt.figure()
    plt.matshow(board)

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, 'wythoff_board.png'))

    if plot:
        # plt.show()
        plt.pause(0.01)
        plt.close()


def plot_wythoff_expected_values(m, n, model, plot=False, path=None):
    """Plot EVs"""
    values = expected_value(m, n, model)

    # !
    fig = plt.figure()
    plt.matshow(values)

    # Save an image?
    if path is not None:
        plt.savefig(os.path.join(path, 'wythoff_expected_values.png'))

    if plot:
        # plt.show()
        plt.pause(0.01)
        plt.close()


def expected_value(m, n, model):
    """Estimate the expected value of each board position"""
    # Build the matrix a values for each (i, j) board
    values = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            board = np.zeros((m, n))
            board[i, j] = 1.0

            values[i, j] = np.max(
                model(Tensor(board.reshape(m * n))).detach().numpy())

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


def stumbler(path,
             num_trials=10,
             epsilon=0.1,
             gamma=0.8,
             learning_rate=0.1,
             wythoff_name='Wythoff3x3',
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
    x, y, board = env.reset()
    env.close()
    m, n = board.shape

    # Create a model of the right size
    model = OneLinQN(m * n, len(possible_actions))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # -------------------------------------------
    # Run some trials
    for trial in range(num_trials):
        x, y, board = env.reset()
        board = Tensor(board.reshape(m * n))

        steps = 0
        while True:
            # -------------------------------------------
            # TODO
            # env.render()

            # -------------------------------------------
            # Look at the world and approximate its value.
            Qs = model(board).squeeze()

            # Make a decision.
            action_index = epsilon_greedy(Qs, epsilon)
            action = possible_actions[int(action_index)]

            Q = Qs[int(action_index)]

            (x, y, next_board), reward, done, _ = env.step(action)
            next_board = Tensor([next_board.reshape(m * n)])

            # Update move counter
            steps += 1

            # ---
            # Learn w/ SGD
            max_Q = model(next_board).detach().max()
            next_Q = reward + (gamma * max_Q)
            loss = F.smooth_l1_loss(Q, next_Q)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # retain is needed for opp. WHY?
            optimizer.step()

            # Shuffle state notation
            board = next_board

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

            Q = Qs[int(action_index)].squeeze()

            (x, y, next_board), reward, done, info = env.step(action)
            next_board = Tensor([next_board.reshape(m * n)])

            # Flip signs so opp victories are punishments
            if reward > 0:
                reward *= -1

            # -
            if log:
                writer.add_scalar(os.path.join(path, 'opp_Q'), Q, trial)
                writer.add_scalar(
                    os.path.join(path, 'opp_reward'), reward, trial)

            # ---
            # Learn from your opponent
            max_Q = model(next_board).detach().max()
            next_Q = reward + (gamma * max_Q)
            loss = F.smooth_l1_loss(Q, next_Q)

            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Plot?
            if (trial % 10) == 0 and log:
                plot_wythoff_expected_values(m, n, model, path=path)

                writer.add_image(
                    'expected_value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_expected_values.png')))

            if done:
                break

        # save min loss for this trial
        if log:
            writer.add_scalar(os.path.join(path, 'error'), loss.data[0], trial)

    # Cleanup and end
    if log:
        writer.close()
        f.close()

    return expected_value(m, n, model)


def wythoff_2(path,
              num_trials=10,
              epsilon=0.1,
              gamma=0.8,
              learning_rate=0.1,
              strategy_name='hot_cold',
              wythoff_name='Wythoff3x3',
              seed=None):
    """A reimplementation of 

     Muyesser, N.A., Dunovan, K. & Verstynen, T., 2018. Learning model-based 
     strategies in simple environments with hierarchical q-networks. 
     pp.1–29. Available at: http://arxiv.org/abs/1801.06689.
     """
    # -------------------------------------------
    # Create path
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # -------------------------------------------
    # Log setup    

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
    x, y, board = env.reset()
    env.close()
    m, n = board.shape

    # Create a model of the right size
    model = OneLinQN(m * n, len(possible_actions))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # -------------------------------------------
    # Run some trials
    for trial in range(num_trials):
        x, y, board = env.reset()
        board = Tensor(board.reshape(m * n))

        steps = 0
        while True:
            # -------------------------------------------
            # Look at the world and approximate its value.
            Qs = model(board).squeeze()

            # Make a decision.
            action_index = epsilon_greedy(Qs, epsilon)
            action = possible_actions[int(action_index)]

            Q = Qs[int(action_index)]

            (x, y, next_board), reward, done, _ = env.step(action)
            next_board = Tensor([next_board.reshape(m * n)])

            # Update move counter
            steps += 1

            # ---
            # Learn w/ SGD
            max_Q = model(next_board).detach().max()
            next_Q = reward + (gamma * max_Q)
            loss = F.smooth_l1_loss(Q, next_Q)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # retain is needed for opp. WHY?
            optimizer.step()

            # Shuffle state notation
            board = next_board

            # -------------------------------------------
            # Log results
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

            Q = Qs[int(action_index)].squeeze()

            (x, y, next_board), reward, done, info = env.step(action)
            next_board = Tensor([next_board.reshape(m * n)])

            # Flip signs so opp victories are punishments
            if reward > 0:
                reward *= -1

            # -
            writer.add_scalar(os.path.join(path, 'opp_Q'), Q, trial)
            writer.add_scalar(os.path.join(path, 'opp_reward'), reward, trial)

            # ---
            # Learn from your opponent
            max_Q = model(next_board).detach().max()
            next_Q = reward + (gamma * max_Q)
            loss = F.smooth_l1_loss(Q, next_Q)

            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Plot?
            if (trial % 10) == 0:
                plot_wythoff_expected_values(m, n, model, path=path)

                writer.add_image(
                    'expected_value',
                    skimage.io.imread(
                        os.path.join(path, 'wythoff_expected_values.png')))

            if done:
                break

        # save min loss for this trial
        writer.add_scalar(os.path.join(path, 'error'), loss.data[0], trial)

    # Cleanup and end
    writer.close()
    f.close()
