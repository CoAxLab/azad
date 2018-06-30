import torch


def greedy(x, index=None):
    """Pick the biggest"""
    # Filter x using index, but first ensure we can
    # map the action back to x' orignal 'space'
    if index is not None:
        x = x[index]

    action = torch.argmax(x).unsqueeze(0)
    action = int(action)

    # Map back to x's original space
    if index is not None:
        action = index[action]

    return action


def epsilon_greedy(x, epsilon, index=None):
    """Pick the biggest, with probability epsilon"""

    # Filter x using index, but first ensure we can
    # map the action back to x' orignal 'space'
    if index is not None:
        x = x[index]

    if torch.rand(1) > epsilon:
        action = torch.argmax(x).unsqueeze(0)
    else:
        action = torch.randint(0, x.shape[0], (1, ))
    action = int(action)

    # Map back to x's original space
    if index is not None:
        action = index[action]

    return action
