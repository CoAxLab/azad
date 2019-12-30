import random
import torch.nn as tnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn

import numpy as np

from azad.exp.alternatives.mcts import MCTS
from azad.exp.alternatives.mcts import MoveCount
from azad.exp.alternatives.mcts import shift_player
from azad.exp.alternatives.mcts import random_policy
from azad.exp.alternatives.mcts import OptimalCount

from azad.exp.alternatives.mcts import MoveCount
from azad.exp.alternatives.mcts import HistoryMCTS
from azad.exp.alternatives.mcts import OptimalCount
from azad.exp.alternatives.mcts import random_policy
from azad.exp.alternatives.mcts import shift_player
from azad.exp.wythoff import save_monitored
from azad.exp.wythoff import create_env
from azad.exp.wythoff import peek
from azad.exp.wythoff import create_all_possible_moves
from azad.local_gym.wythoff import locate_cold_moves
from azad.local_gym.wythoff import locate_all_cold_moves


class ReplayBuffer(object):
    """A Replay buffer, adapted from ..."""
    def __init__(self, window_size=1e6, batch_size=4096):
        self.window_size = int(window_size)
        self.batch_size = batch_size
        self.buffer = []

    def update(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample(self):
        # Sample uniformly across positions.
        move_sum = float(sum(len(g) for g in self.buffer))
        games = np.random.choice(self.buffer,
                                 size=self.batch_size,
                                 p=[len(g) / move_sum for g in self.buffer])
        game_pos = [(g, np.random.randint(len(g))) for g in games]
        return [g[i] for (g, i) in game_pos]


class Network(nn.Module):
    """An MLP Alphazero-alike network."""
    def __init__(self, in_channels=2, num_hidden1=100, num_hidden2=25):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.head1 = nn.Linear(num_hidden2, 1)
        self.head2 = nn.Linear(num_hidden2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head1(x), self.head2


def train(network, memory):
    pass
