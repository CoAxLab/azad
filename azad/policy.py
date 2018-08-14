import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


def _np_epsilon_greedy(x, epsilon, index=None):
    # Filter x using index, but first ensure we can
    # map the action back to x' orignal 'space'
    if index is not None:
        x = x[index]

    if np.random.rand() < epsilon:
        action = np.random.randint(0, x.shape[0])
    else:
        action = np.argmax(x)

    # Map back to x's original space
    if index is not None:
        action = index[action]

    return action


def _th_epsilon_greedy(x, epsilon, index=None):
    # Filter x using index, but first ensure we can
    # map the action back to x' orignal 'space'
    if index is not None:
        x = x[index]

    if torch.rand(1) < epsilon:
        action = torch.randint(0, x.shape[0], (1, ))
    else:
        action = torch.argmax(x).unsqueeze(0)

    action = int(action)

    # Map back to x's original space
    if index is not None:
        action = index[action]

    return action


def epsilon_greedy(x, epsilon, index=None, mode='pytorch'):
    """Pick the biggest, with probability epsilon"""

    if mode == 'pytorch':
        return _th_epsilon_greedy(x, epsilon, index=index)
    elif mode == 'numpy':
        return _np_epsilon_greedy(x, epsilon, index=index)
    else:
        raise ValueError("mode must be numpy or pytorch")


def softmax(x, beta=0.98, index=None):
    """Softmax policy"""
    # Filter x using index, but first ensure we can
    # map the action back to x' orignal 'space'
    if index is not None:
        x = x[index]

    probs = F.softmax(x * beta)
    m = Categorical(probs)
    action = m.sample()

    # Map back to x's original space
    if index is not None:
        action = index[action]

    return action
