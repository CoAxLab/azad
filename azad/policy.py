import torch


def ep_greedy(x, ep, dim):
    if torch.rand(1) > ep:
        action = torch.argmax(x, dim)
    else:
        action = torch.randint(0, x.shape[dim], (1, ))

    return action
