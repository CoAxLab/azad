#!/usr/bin/env python3
"""Run azad experiments"""
import fire
import torch
import numpy as np


def exp_build():
    raise NotImplementedError("TODO.")


def exp_list():
    """List all registered experiments"""
    # Loop over all run in this submodule
    # if the fn name is exp_INT print its name
    # and print its docstring
    raise NotImplementedError("TODO.")


def exp_1(n_epochs=10):
    """Train DQN on a pole cart"""

    # Load a DQN,
    # the cart from the gym

    # Train and test them... 

    return None


if __name__ == "__main__":
    fire.Fire({
        "exp_list": exp_list,
        "exp_build": exp_build,
        "exp_1": exp_1,
    })
