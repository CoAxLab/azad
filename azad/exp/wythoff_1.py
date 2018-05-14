import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import gym
from gym import wrappers
import azad.local_gym

from azad.stumblers import OneLinQN
from azad.policy import epsilon_greedy
from azad.util import ReplayMemory

import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Handle dtypes for the device
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor

# ---------------------------------------------------------------


def wythoff_1(name,
              num_trials=10,
              epsilon=0.1,
              gamma=0.8,
              learning_rate=0.01,
              wythoff_name='Wythoff3x3',
              seed=None):
    """Train a Q-agent to play Wythoff's game, using SGD."""

    # Valid moves (in this simplified instantiation)
    possible_actions = [(-1, 0), (0, -1), (-1, -1)]

    # -------------------------------------------
    # The world is a cart....
    env = gym.make('{}-v0'.format(wythoff_name))
    env = wrappers.Monitor(
        env, './tmp/{}-v0-1'.format(wythoff_name), force=True)

    board_size = (env.m, env.n)

    # -------------------------------------------
    # Seeding...
    env.seed(seed)
    np.random.seed(seed)

    # -------------------------------------------
    # Build a Q agent, its memory, and its optimizer
    model = OneLinQN(board_size, len(possible_actions))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # -------------------------------------------
    # Run some trials
    trials = []
    trial_steps = []
    trial_rewards = []
    trial_values = []
    trial_actions = []

    for trial in range(num_trials):
        state = Tensor(env.reset())
        steps = 0
        while True:
            # -------------------------------------------
            # env.render()

            # -------------------------------------------
            # Look at the world and approximate its value.
            Qs = model(state)

            # Make a decision.
            action_index = epsilon_greedy(Q, epsilon)
            action = possible_actions[action_index]

            Q = Qs[int(action_index)]

            next_state, reward, done, _ = env.step(action)
            next_state = Tensor([next_state])

            # Update move counter
            steps += 1

            # ---
            # Learn w/ SGD
            max_Q = model(next_state).detach().max()
            next_Q = reward + (gamma * max_Q)
            loss = F.smooth_l1_loss(Q, next_Q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Shuffle state notation
            state = next_state

            # -------------------------------------------
            # Save results
            trials.append(trial)
            trial_steps.append(steps)
            trial_rewards.append(float(reward))
            trial_actions.append(int(action))
            trial_values.append(float(Q))

            # -------------------------------------------
            # If the game is over, stop
            if done:
                break

            # -------------------------------------------
            # Otherwise the opponent moves
            action_index = np.random.randint(0, len(possible_actions))
            action = possible_actions[action_index]

            Q = Qs[int(action_index)]

            next_state, reward, done, _ = env.step(action)

            # Flip signs so opp victories are punishments
            if reward > 0:
                reward *= -1

            # ---
            # Learn from your opponent
            max_Q = model(next_state).detach().max()
            next_Q = reward + (gamma * max_Q)
            loss = F.smooth_l1_loss(Q, next_Q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                break

    results = list(
        zip(trials, trial_steps, trial_actions, trial_rewards, trial_values))
    return results