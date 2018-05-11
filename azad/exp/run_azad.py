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
from azad.exp import exp_1

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


if __name__ == "__main__":
    fire.Fire({
        "exp_list": exp_list,
        "exp_build": exp_build,
        "exp_1": exp_1,
    })
