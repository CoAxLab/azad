from collections import defaultdict
from csv import DictReader
import scipy.stats as st
import numpy as np


def load_params(file):
    """Custom param loading fn"""
    table = defaultdict(list)
    with open(file, 'r') as f:
        reader = DictReader(f)
        for row in reader:
            for col, dat in row.items():
                table[col].append(float(dat))
    return table


def load_evaluated(files, game_code):
    """Load tabular data from an evaulation"""
    table = defaultdict(list)

    for i, fi in enumerate(files):
        fi_name = fi.split('/')[-1].split('.')[0]

        with open(fi, 'r') as f:
            reader = DictReader(f)
            for row in reader:
                table["game_code"].append(game_code)
                table["file_index"].append(i)
                table["file_name"].append(fi_name)

                for col, dat in row.items():
                    table[col].append(float(dat))

    return table


def load_monitored(file):
    """Load monitored data."""

    table = defaultdict(list)
    file_name = file.split('/')[-1].split('.')[0]

    with open(file, 'r') as f:
        reader = DictReader(f)
        for row in reader:
            table["file"].append(file_name)

            for col, dat in row.items():
                table[col].append(float(dat))

    return table


def join_monitored(files, sort_key='episode'):
    """Join several monitored datasets, by column."""

    joined = load_monitored(files[0])

    for fi in files[1:]:
        for k, v in load_monitored(fi).items():
            joined[k].extend(v)

    if sort_key is not None:
        idx = np.argsort(joined[sort_key]).tolist()

        for k, v in joined.items():
            joined[k] = [v[i] for i in idx]

    return joined


def score_summary(exp, ):
    """Summarize lists of exps"""
    t = len(exp[0]["score"])
    l = len(exp)
    X = np.zeros((t, l))

    for i, mon in enumerate(exp):
        X[:, i] = mon["score"]

    return exp[0]["episode"], X.mean(1), st.sem(X, axis=1)