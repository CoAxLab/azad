import numpy as np
from itertools import product


def create_grid(name, fmt=None, **kwargs):
    """Create a .csv for doing a grid param search."""

    keys = sorted(list(kwargs.keys()))
    values = []
    for k in keys:
        start, stop, n = kwargs[k]
        v = np.linspace(start, stop, n)
        values.append(v)

    table = product(*values)
    i_table = []
    for i, t in enumerate(table):
        i_table.append((i, *t))
    table = np.vstack(i_table)

    head = "row_code," + ",".join(keys)
    if fmt is None:
        fmt = '%i,' + '%.6f,' * (len(keys) - 1) + '%.6f'
    np.savetxt(name, table, delimiter=",", header=head, fmt=fmt, comments="")


def create_random(name, fmt=None, seed_value=None, **kwargs):
    """Create a .csv for doing a random param search."""
    np.random.seed(seed_value)

    keys = sorted(list(kwargs.keys()))
    values = []
    for k in keys:
        start, stop, n = kwargs[k]
        v = np.random.uniform(start, stop, n)
        values.append(v)

    table = product(*values)
    i_table = []
    for i, t in enumerate(table):
        i_table.append((i, *t))
    table = np.vstack(i_table)

    head = "row_code," + ",".join(keys)
    if fmt is None:
        fmt = '%i,' + '%.6f,' * (len(keys) - 1) + '%.6f'
    np.savetxt(name, table, delimiter=",", header=head, fmt=fmt, comments="")
