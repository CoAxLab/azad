import torch


def greedy(x):
    """Pick the biggest"""
    action = torch.argmax(x).unsqueeze(0)

    # Ensure type consistency... 
    action = action.type(torch.FloatTensor)

    return action


def epsilon_greedy(x, epsilon):
    """Pick the biggest, with probability epsilon"""
    if torch.rand(1) > epsilon:
        action = torch.argmax(x).unsqueeze(0)
    else:
        action = torch.randint(0, x.shape[0], (1, ))

    # Ensure type consistency... 
    action = action.type(torch.FloatTensor)

    return action
