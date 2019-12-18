import random
import torch.nn as tnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn


class ReplayMemory(object):
    """A very generic memory system, with a finite capacity."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size <= len(self.memory):
            return self.memory
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReLu2(nn.Module):
    """2-layer ReLu Network"""
    def __init__(self, in_channels, num_actions, num_hidden=200):
        super(ReLu2, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class ReLu3(nn.Module):
    """3-layer ReLu Network"""
    def __init__(self,
                 in_channels,
                 num_actions,
                 num_hidden1=200,
                 num_hidden2=100):
        super(ReLu3, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DeepTable3(nn.Module):
    """A deep differentialable 'Table' for learning one-hot input and output.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hidden1=200,
                 num_hidden2=100):
        super(DeepTable3, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1, bias=False)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2, bias=False)
        self.fc3 = nn.Linear(num_hidden2, out_channels, bias=False)

        self.fc1.weight.data.uniform_(0.0, 0.0)
        self.fc3.weight.data.uniform_(0.0, 0.0)
        # self.fc3.bias.data.uniform_(0, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(self.fc2(x))
        return self.fc3(x)


class Table(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        """A differentialable 'Table' for learning one-hot input and output."""

        super(Table, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels, bias=False)
        self.fc1.weight.data.uniform_(0, 0)

    def forward(self, x):
        return self.fc1(x)


class LinQN1(nn.Module):
    def __init__(self, in_channels=4, num_actions=2):
        """One layer linear Q model.
        
        Note: this functions as a Q-agent in the Sutton and 
        Barto sense, wanted a torch optim compatible implementation.
        ...We waste some electrons to simplify the code...
        """

        super(LinQN1, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_actions)

    def forward(self, x):
        return self.fc1(x)


class HotCold2(nn.Module):
    """Layers for a Hot-Cold strategy
    
    As described in:
    
    Muyesser, N.A., Dunovan, K. & Verstynen, T., 2018. Learning model-based 
    strategies in simple environments with hierarchical q-networks. , pp.1–29. A
    vailable at: http://arxiv.org/abs/1801.06689.
    """
    def __init__(self, in_channels=2, num_hidden1=15):
        super(HotCold2, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return self.fc2(x)


class HotCold3(nn.Module):
    """Two layers for a Hot-Cold strategy
    
    Related to the model described in:
    
    Muyesser, N.A., Dunovan, K. & Verstynen, T., 2018. Learning model-based 
    strategies in simple environments with hierarchical q-networks. , pp.1–29. A
    vailable at: http://arxiv.org/abs/1801.06689.
    """
    def __init__(self, in_channels=2, num_hidden1=100, num_hidden2=25):
        super(HotCold3, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return self.fc3(x)
