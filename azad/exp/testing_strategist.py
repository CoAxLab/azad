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

from azad.exp.wythoff import *


def testing_strategist(path,
                       num_trials=1000,
                       learning_rate=0.1,
                       strategic_default_value=0.5,
                       wythoff_name_stumbler='Wythoff50x50',
                       wythoff_name_strategist='Wythoff50x50',
                       log=False,
                       seed=None):

    # -------------------------------------------
    # Setup
    # -------------------------------------------
    # Create path
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    env = gym.make('{}-v0'.format(wythoff_name_strategist))
    env = wrappers.Monitor(
        env, './tmp/{}-v0-1'.format(wythoff_name_strategist), force=True)

    possible_actions = [(-1, 0), (0, -1), (-1, -1)]

    # Train params
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
    m, n, board = peak(wythoff_name_strategist)
    o, p, _ = peak(wythoff_name_stumbler)

    # Create a model, of the right size.
    model = HotCold(2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    memory = ReplayMemory(10000)

    # Run learning trials. The 'stumbler' is just the opt
    # cold board
    for trial in range(num_trials):
        strategic_value = create_cold_board(o, p)
        # strategic_value = (strategic_value + 1.0) / 2.0
        strategic_value = pad_board(m, n, strategic_value,
                                    strategic_default_value)

        # ...Into tuples
        s_data = convert_ijv(strategic_value)

        # Filter values great/eq to defualt
        # s_filtered = []
        # for (c, v) in s_data:
        #     if v >= strategic_default_value:
        #         continue
        #         print("Dropping {}".format(v))
        #     else:
        #         s_filtered.append([c, v])

        for d in s_data:
            memory.push(*d)

        # Train the strategist on perfect data
        coords = []
        values = []
        samples = memory.sample(batch_size)

        for c, v in samples:
            coords.append(c)
            values.append(v)

        coords = torch.tensor(
            np.vstack(coords), requires_grad=True, dtype=torch.float)
        values = torch.tensor(values, requires_grad=True, dtype=torch.float)

        # Scale coords
        coords = coords / m

        # Making some preditions,
        predicted_values = model(coords).detach().squeeze()

        # and find their loss.
        # loss = F.smooth_l1_loss(values, predicted_values)
        # loss = F.mse_loss(values, predicted_values)
        # loss = F.mse_loss(
        #     torch.clamp(values, 0, 1), torch.clamp(predicted_values, 0, 1))
        loss = F.binary_cross_entropy(values, predicted_values)

        # Walk down the hill of righteousness!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compare strategist and stumbler. Count strategist wins.
        # win = evaluate_models(stumbler_model, model, stumbler_env, env)

        # Use the trained strategist to generate a bias_board,
        bias_board = estimate_strategic_value(m, n, model)

        if log:
            writer.add_scalar(os.path.join(path, 'error'), loss.data[0], trial)

            plot_wythoff_board(
                strategic_value, path=path, name='strategy_board.png')
            writer.add_image(
                'Expected value board',
                skimage.io.imread(os.path.join(path, 'strategy_board.png')))

            plot_wythoff_board(bias_board, path=path, name='bias_board.png')
            writer.add_image(
                'Strategist learning',
                skimage.io.imread(os.path.join(path, 'bias_board.png')))

    # The end
    if log:
        writer.close()

    return model, env
