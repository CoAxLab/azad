#!/usr/bin/env python3
"""Run azad experiments"""
import fire
import torch

from math import exp

import numpy as np

import gym
from gym import wrappers

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from azad.stumblers import TwoQN
from azad.stumblers import ThreeQN
from azad.stumblers import DQN
from azad.policy import ep_greedy
from azad.util import ReplayMemory
from azad.util import plot_cart_durations

import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Handle dtypes for the device
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor

# ---------------------------------------------------------------


def exp_build():
    raise NotImplementedError("TODO.")


def exp_list():
    """List all registered experiments"""
    # Loop over all run in this submodule
    # if the fn name is exp_INT print its name
    # and print its docstring
    raise NotImplementedError("TODO.")


def exp_1(name,
          num_episodes=500,
          epsilon=0.1,
          epsilon_min=0.01,
          epsilon_tau=500,
          gamma=1,
          learning_rate=0.001,
          num_hidden=200,
          batch_size=64):
    """Train DQN on a pole cart"""

    # -------------------------------------------
    # The world is a cart....
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, './tmp/cartpole-v0-1', force=True)

    # -------------------------------------------
    # Init the DQN, it's memory, and its optim
    # model = ThreeQN(4, 2, num_hidden1=1000, num_hidden2=200)
    model = TwoQN(4, 2, num_hidden=num_hidden)
    memory = ReplayMemory(10000)
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # -------------------------------------------
    # Run some episodes
    episode_durations = []

    for episode in range(num_episodes):
        state = Tensor(env.reset())

        steps = 0
        while True:
            env.render()

            # -------------------------------------------
            # Look at the world and approximate its value.
            Q = model(state)

            # Make a decision.
            epsilon_step = epsilon_min + (epsilon - epsilon_min) * exp(
                -1.0 * steps / epsilon_tau)
            action = ep_greedy(Q, epsilon_step)
            next_state, reward, done, _ = env.step(int(action))

            # Punishment, at the end of the world.
            if done:
                reward = -1

            # Always remember the past
            # (you are still doomed to repeat it).
            next_state = Tensor(next_state)
            reward = Tensor([reward])

            memory.push(
                state.unsqueeze(0),
                action.unsqueeze(0),
                next_state.unsqueeze(0), reward.unsqueeze(0))

            # -------------------------------------------
            # Learn from the last result. 

            # If there is not enough in memory, 
            # don't try and learn anything.
            if done:
                print(">>> {2} Episode {0} finished after {1} steps".format(
                    episode, steps, '\033[92m'
                    if steps >= 195 else '\033[99m'))

                episode_durations.append(steps)
                plot_cart_durations(episode_durations)

                break
            elif len(memory) < batch_size:
                continue

            # Grab some examples from memory
            # and repackage them.
            transitions = memory.sample(batch_size)
            t_states, t_actions, t_next_states, t_rewards = zip(*transitions)

            # Conversions....
            t_states = Variable(torch.cat(t_states))
            t_actions = Variable(torch.cat(t_actions))
            t_rewards = Variable(torch.cat(t_rewards)).squeeze()
            t_next_states = Variable(torch.cat(t_next_states))

            # Possible Qs for actions
            Qs = model(t_states).gather(
                1, t_actions.type(torch.LongTensor)).squeeze()

            # In Q learning we use the max Q of the next state,
            # and the reward, to estimate future Qs value
            max_Qs = model(t_next_states).detach().max(1)[0]
            future_Qs = t_rewards + (gamma * max_Qs)

            # Want to min the loss between predicted Qs
            # and the observed
            loss = F.smooth_l1_loss(Qs, future_Qs)

            # Grad. descent!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -------------------------------------------
            state = next_state
            steps += 1

            if done:
                break

    # -------------------------------------------
    # Clean up
    env.env.close()
    plt.ioff()
    plt.savefig("{}.png".format(name))
    plt.close()

    return episode_durations


if __name__ == "__main__":
    fire.Fire({
        "exp_list": exp_list,
        "exp_build": exp_build,
        "exp_1": exp_1,
    })
