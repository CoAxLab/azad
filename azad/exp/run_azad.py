#!/usr/bin/env python3
"""Run azad experiments"""
import fire

from azad.exp import exp_1


def exp_build():
    raise NotImplementedError("TODO.")


def exp_list(details=False):
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
