import random
import torch
import torch.nn as tnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn

import numpy as np

from copy import deepcopy

from azad.exp.alternatives.mcts import Node
from azad.exp.alternatives.mcts import MCTS
from azad.exp.alternatives.mcts import HistoryMCTS
from azad.exp.alternatives.mcts import MoveCount
from azad.exp.alternatives.mcts import OptimalCount
from azad.exp.alternatives.mcts import shift_player
from azad.exp.alternatives.mcts import random_policy
from azad.exp.alternatives.mcts import random_policy
from azad.exp.alternatives.mcts import shift_player
from azad.exp.wythoff import save_monitored
from azad.exp.wythoff import create_env
from azad.exp.wythoff import peek
from azad.exp.wythoff import create_all_possible_moves
from azad.local_gym.wythoff import locate_cold_moves
from azad.local_gym.wythoff import locate_all_cold_moves
from azad.local_gym.wythoff import create_board


class AlphaZero(MCTS):
    """MCTS, with state generalization."""
    def expand(self, node, available, network, env, device="cpu"):
        """Expand the tree with a random new action, which should be valued
        by a rollout.
        """
        if len(node.children) > 0:
            raise ValueError("expand called wrongly")

        # Create input
        all_moves = network.all_moves
        board = create_board(env.x, env.y, env.m, env.n)

        priors, value = network(
            torch.tensor(board, device=device).unsqueeze(0).float())

        # Find new candidate actions, and create Nodes for each.
        # This will leave them available for select later.
        for a in available:
            new = Node(name=a,
                       initial_count=1,
                       initial_value=0,
                       prior=float(priors[0, all_moves.index(a)].item()))
            node.add(new)  # inplace update

        # Pick a move to rollout, and add it to the path.
        move = self.default_policy(available)
        self.path.append(node.children[available.index(move)])

        return move, node, value


class ConvBlock(nn.Module):
    def __init__(self, board_size):
        super(ConvBlock, self).__init__()
        self.action_size = board_size
        self.conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, x):
        # batch_size x channels x board_x x board_y
        x = x.view(-1, 1, self.action_size, self.action_size)
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
        self.fc = nn.Linear(self.board_size**2 * 32, self.board_size**2)

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
    def __init__(self, board_size=15):
        super(ResNet, self).__init__()

        self.board_size = board_size
        self.all_moves = create_all_possible_moves(board_size, board_size)
        if len(self.all_moves) != self.board_size**2:
            raise ValueError("moves and board don't match")

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


class MLP(nn.Module):
    """3-layer MLP for AlphaZero"""
    def __init__(self, board_size=15, num_hidden1=2000, num_hidden2=1000):
        super(MLP, self).__init__()
        # Init
        self.board_size = board_size
        self.all_moves = create_all_possible_moves(board_size, board_size)
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2

        # Sanity between board and action space
        if len(self.all_moves) != self.board_size**2:
            raise ValueError("moves and board don't match")

        # Shared layers
        self.fc1 = nn.Linear(self.board_size**2, self.num_hidden1)
        self.fc2 = nn.Linear(self.num_hidden1, self.num_hidden2)

        # Policy head
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc3 = nn.Linear(self.num_hidden2, self.board_size**2)

        # Value head
        self.fc4 = nn.Linear(self.num_hidden2, 1)

    def forward(self, x):
        x = x.view(-1, self.board_size**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        p = F.relu(self.fc3(x))
        p = self.logsoftmax(p).exp()
        v = torch.tanh(self.fc4(x))

        return p, v


def train_resnet(network, memory, optimizer, batch_size, clip_grad=False):
    """Train the value approx network, a resnet"""

    # Sample and vectorize the samples
    samples = memory.sample(batch_size)
    actions_i, boards, ps, values = zip(*samples)

    actions_i = torch.stack(actions_i).squeeze().unsqueeze(1)
    boards = torch.stack(boards)
    ps = torch.stack(ps).squeeze().unsqueeze(1)
    values = torch.stack(values).squeeze().unsqueeze(1)

    # Get values and p from the resnet
    ps_hat, values_hat = network(boards)
    ps_hat = ps_hat.gather(1, actions_i.long())

    # Calc loss, in two parts. Value and policy.
    value_error = (values - values_hat)**2
    policy_error = torch.sum((-ps * (1e-8 + ps_hat.float()).float().log()), 1)

    # Join 'em
    loss = (value_error.view(-1).float() + policy_error).mean()

    # Learn
    optimizer.zero_grad()
    loss.backward()
    if clip_grad:
        for param in network.parameters():
            param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return network, loss


def run_alphazero(player,
                  env,
                  network,
                  num_simulations=10,
                  c=1.41,
                  max_size=500,
                  default_policy=random_policy,
                  mcts=None,
                  device='cpu'):

    if mcts is None:
        mcts = AlphaZero(c=c, default_policy=default_policy)

    for _ in range(num_simulations):
        # Reinit
        node = mcts.reset()
        scratch_player = deepcopy(player)
        scratch_env = deepcopy(env)
        done = False

        # Select
        while mcts.expanded(node) and not done:
            node = mcts.select(node)
            _, reward, done, _ = scratch_env.step(node.name)
            scratch_player = shift_player(scratch_player)

        # Expand, if we are not terminal.
        if not done:
            move, node, value = mcts.expand(node,
                                            scratch_env.moves,
                                            network,
                                            scratch_env,
                                            device=device)
            _, reward, done, _ = scratch_env.step(move)
        if not done:
            reward = value

        # Learn
        if scratch_player == player:
            mcts.backpropagate(scratch_player, reward)
        else:
            mcts.backpropagate(scratch_player, -1 * reward)

    return mcts.best(), mcts
