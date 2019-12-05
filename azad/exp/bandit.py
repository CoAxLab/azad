import os
import errno

import torch
import torch.optim as optim
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import wrappers
import azad.local_gym

from azad.models import LinQN1
from azad.policy import epsilon_greedy

# ---------------------------------------------------------------
# Handle dtypes for the device
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor

# ---------------------------------------------------------------


def bandit_stumbler(path,
                    num_trials=10,
                    epsilon=0.1,
                    gamma=0.8,
                    learning_rate=0.1,
                    log_path=None,
                    bandit_name='BanditTwoArmedDeterministicFixed'):
    """Train a Q-agent to play n-bandit, using SGD.
    
    Note: bandits are drawm from azad.local_gym. See that module for
    more information on the bandits.
    """
    # Create path
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # -------------------------------------------
    # setup

    # Logging
    if log_path is None:
        log_path = path
    writer = SummaryWriter(log_dir=log_path)

    # The world is a slot machine!
    env = gym.make('{}-v0'.format(bandit_name))
    env = wrappers.Monitor(
        env, './tmp/{}-v0-1'.format(bandit_name), force=True)

    # Init the 'agent'
    model = LinQN1(1, 2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # -------------------------------------------
    # Run some trials

    # Loop over trials not batchs, doing
    # SGD on each outcome
    # (no idea how well this will work)
    for trial in range(num_trials):
        state = Tensor([env.reset()])

        if trial == 0:
            writer.add_graph(model, state)

        # Look at the world and approximate its value then act.
        Qs = model(state)
        action = epsilon_greedy(Qs, epsilon)

        Q = Qs[int(action)]

        next_state, reward, _, _ = env.step(int(action))
        next_state = Tensor([next_state])

        # Walk down the hill o' rightenous!
        max_Q = model(next_state).detach().max()
        next_Q = reward + (gamma * max_Q)
        loss = F.smooth_l1_loss(Q, next_Q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        for path, param in model.named_parameters():
            writer.add_histogram(path, param.clone().cpu().data.numpy(), trial)
        writer.add_scalar(os.path.join(log_path, 'error'), loss.data[0], trial)
        writer.add_scalar(os.path.join(log_path, 'Q'), Q, trial)
        writer.add_scalar(os.path.join(log_path, 'reward'), reward, trial)
        writer.add_scalar(os.path.join(log_path, 'state'), state, trial)

    # Cleanup and end
    writer.close()

    return model, env