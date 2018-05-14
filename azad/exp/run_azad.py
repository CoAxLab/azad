#!/usr/bin/env python3
"""Run azad experiments"""
import fire

from azad.exp import cart_1
from azad.exp import bandit_1


def exp_build():
    raise NotImplementedError("TODO.")


def exp_list(details=False):
    """List all registered experiments"""
    # Loop over all run in this submodule
    # if the fn name is exp_INT print its name
    # and print its docstring
    raise NotImplementedError("TODO.")


# TODO: autobuild this list for all exps.*
if __name__ == "__main__":
    fire.Fire({
        "exp_list": exp_list,
        "exp_build": exp_build,
        "cart_1": cart_1,
        "bandit_1": bandit_1
    })
