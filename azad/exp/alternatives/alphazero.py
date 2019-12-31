import random
import torch
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


class ConvBlock(nn.Module):
    def __init__(self, board_size):
        super(ConvBlock, self).__init__()
        self.action_size = board_size
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, x):
        # batch_size x channels x board_x x board_y
        x = s.view(-1, 1, self.board_size, self.board_size)
        x = F.relu(self.bn1(self.conv1(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class Head(nn.Module):
    def __init__(self, board_size=500):
        super(Head, self).__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(128, 3, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * self.board_size**2, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(self.board_size**2 * 32, 7)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 3 *
                   self.board_size**2)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, self.board_size**2 * 32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class ResNet(nn.Module):
    """An AlphaZero-esque ResNet, configured for Wythoffs game and its
    analogs.
    """
    def __init__(self, board_size=500):
        super(ResNet, self).__init__()
        self.board_size = board_size
        self.conv1 = ConvBlock(board_size=self.board_size)
        for block in range(19):
            setattr(self, "res%i" % block, ResBlock())
        self.head1 = Head(board_size=self.board_size)

    def forward(self, x):
        x = self.conv1(x)
        for block in range(19):
            x = getattr(self, "res%i" % block)(x)
        p, v = self.head1(x)
        return p, v


def train(network, memory, optimizer):
    pass
