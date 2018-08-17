#!/usr/bin/env python3
"""Run azad experiments"""
import fire

from azad import exp as a_exps

if __name__ == "__main__":
    # Auto build a CL API from all azad.exps.
    cl = {}

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
