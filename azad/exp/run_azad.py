#!/usr/bin/env python3
"""Run azad experiments"""
import fire
import torch
import numpy as np

import gym
from gym import wrappers

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from azad.stumblers import DQN
from azad.policy import ep_greedy
from azad.util import ReplayMemory

import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Handle dtypes for the device
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor


def exp_build():
    raise NotImplementedError("TODO.")


def exp_list():
    """List all registered experiments"""
    # Loop over all run in this submodule
    # if the fn name is exp_INT print its name
    # and print its docstring
    raise NotImplementedError("TODO.")


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def _exp_1_learn(model, optimizer, memory, gamma, batch_size=64):
    # If there is not enough in memory, 
    # don't try and learn anything.
    if len(memory) < batch_size:
        return None

    # Grab some examples from memory
    # and repackage them.
    transitions = memory.sample(batch_size)
    states, actions, next_states, rewards = zip(*transitions)

    # Conversions....
    states = Variable(torch.cat(states))
    actions = Variable(torch.cat(actions))
    rewards = Variable(torch.cat(rewards))
    next_states = Variable(torch.cat(next_states))

    # Possible Qs for actions
    Qs = model(states).gather(1, actions)

    # In Q learning we use the max Q of the next state,
    # and the reward, to estimate future Qs value
    max_Qs = model(next_states).detach().max(1)[0]
    future_Qs = rewards + (gamma * max_Qs)

    # Want to min the loss between predicted Qs
    # and the observed
    loss = F.smooth_l1_loss(Qs, future_Qs)

    # Grad. descent!
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def exp_1(n_episodes=10, epsilon=0.2, gamma=1, alpha=0.001):
    """Train DQN on a pole cart"""

    # -------------------------------------------
    # The world is a cart....
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, './tmp/cartpole-v0-1')

    # -------------------------------------------
    # Init the DQN, it's memory, and its optim
    model = DQN(4, 2)
    memory = ReplayMemory(10000)
    optimizer = optim.Adam(model.parameters(), alpha)

    # -------------------------------------------
    # Run seperate episodes
    episode_durations = []  # Bigger is better....

    for episode in range(n_episodes):
        state = env.reset()
        steps = 0
        while True:
            env.render()
            action = ep_greedy(FloatTensor([state]), epsilon, 0)
            next_state, reward, done, _ = env.step(action[0, 0])

            # negative reward when attempt ends
            if done:
                reward = -1

            memory.push((
                FloatTensor([state]),
                action,  # action is already a tensor
                FloatTensor([next_state]),
                FloatTensor([reward])))

            _exp_1_learn(memory, optimizer, memory, gamma)

            state = next_state
            steps += 1

            if done:
                print("{2} Episode {0} finished after {1} steps".format(
                    episode, steps, '\033[92m'
                    if steps >= 195 else '\033[99m'))

                episode_durations.append(steps)
                plot_durations(episode_durations)

                break

    return None


if __name__ == "__main__":
    fire.Fire({
        "exp_list": exp_list,
        "exp_build": exp_build,
        "exp_1": exp_1,
    })
