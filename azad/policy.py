import torch


def ep_greedy(x, ep):
    if torch.rand(1) > ep:
        action = torch.argmax(x).unsqueeze(0)
    else:
        action = torch.randint(0, x.shape[0], (1, ), dtype=torch.int)

    return action
