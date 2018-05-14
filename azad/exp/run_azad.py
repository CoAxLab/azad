#!/usr/bin/env python3
"""Run azad experiments"""
import fire

from azad import exp as a_exps
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


if __name__ == "__main__":
    # Auto build a CL API from all azad.exps
    cl = {"exp_list": exp_list, "exp_build": exp_build}

    # Get all the attrs on azad.exp
    all_possible = dir(a_exps)
    for a_poss in all_possible:
        # Skip this program,
        if a_poss == "run_azad":
            continue
        # and any hidden things,
        elif a_poss.startswith("__"):
            continue
        elif a_poss.startswith("_"):
            continue
        # otherwise add.
        else:
            cl[a_poss] = getattr(a_exps, a_poss)

    # Instantiate the CL interface, with Fire!
    fire.Fire(cl)
