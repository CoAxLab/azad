import numpy as np
from itertools import product
from itertools import cycle


def create_grid(name, fmt=None, num_gpu=0, **kwargs):
    """Create a .csv for doing a grid param search."""
    # Sanity
    num_gpu = int(num_gpu)
    if num_gpu < 0:
        raise ValueError("num_gpu must be positive")

    # Create the grid
    keys = sorted(list(kwargs.keys()))
    values = []
    for k in keys:
        start, stop, n = kwargs[k]
        v = np.linspace(start, stop, n)
        values.append(v)

    # then rearrange it into a nice csv/table
    table = product(*values)
    i_table = []

    # The table depends on GPU use, so...
    # No GPU
    if num_gpu == 0:
        for i, t in enumerate(table):
            i_table.append((i, *t))

        head = "row_code," + ",".join(keys)
        if fmt is None:
            fmt = '%i,' + '%.6f,' * (len(keys) - 1) + '%.6f'

    # Use GPU(s). Dividing the work equally between 'em.
    else:
        device_count = cycle(range(num_gpu))
        for i, t in enumerate(table):
            i_table.append((i, next(device_count), *t))

        head = "row_code,device_code," + ",".join(keys)
        if fmt is None:
            fmt = '%i,%i,' + '%.6f,' * (len(keys) - 1) + '%.6f'

    # Form final table and save it.
    table = np.vstack(i_table)
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
