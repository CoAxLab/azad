"""Learn to play Wythoff's with a DQN, using a (x,y) board representation."""
import os, csv
import sys

import errno
import pudb

from collections import defaultdict
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import torch.nn as tnn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

import numpy as np
from scipy.constants import golden

import matplotlib.pyplot as plt
import seaborn as sns

import skimage
from skimage import data, io

import gym
from gym import wrappers
import azad.local_gym
from azad.local_gym.wythoff import create_moves
from azad.local_gym.wythoff import create_all_possible_moves
from azad.local_gym.wythoff import locate_moves
from azad.local_gym.wythoff import create_cold_board
from azad.local_gym.wythoff import create_board
from azad.local_gym.wythoff import cold_move_available
from azad.local_gym.wythoff import locate_closest_cold_move
from azad.local_gym.wythoff import locate_cold_moves
from azad.local_gym.wythoff import locate_all_cold_moves

from azad.models import ReplayMemory
from azad.policy import epsilon_greedy as e_greedy

from azad.exp.wythoff import peek
from azad.exp.wythoff import create_env
from azad.exp.wythoff import create_monitored
from azad.exp.wythoff import flatten_board
from azad.exp.wythoff import plot_wythoff_board
from azad.exp.wythoff import save_monitored
from azad.exp.wythoff import expected_value
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self, m, n, num_actions):
        """Layers for a Deep Q Network

        Based on:
        Minh, V. et al, 2015. Human-level control through deep reinforcement 
        learning. Nature, 518, pp.529–533. Available at: 
        http://dx.doi.org/10.1038/nature14236.
        
        Code modified from:
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        
        Params
        ------
        m,n: int
            Board size 
        num_actions: int 
            Number of action-value to output, one-to-one correspondence 
            to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # With the above and fixed params each conv layer
        # looses n - 2 in size,and there are three layers.
        # So calc the final numel for the linear 'decode'
        # at the end.
        self.fc4 = nn.Linear(64 * (n - (2 * 3)) * (m - (2 * 3)), 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class DQN_mlp(nn.Module):
    """Layers for a Deep Q Network, based on a simple MLP."""
    def __init__(self, m, n, num_actions, num_hidden1=1000, num_hidden2=2000):
        super(DQN_mlp, self).__init__()
        self.m = m
        self.n = n
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2

        self.fc1 = nn.Linear(m * n, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_hidden2)
        self.fc4 = nn.Linear(num_hidden2, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def train_dqn(batch_size,
              model,
              memory,
              optimizer,
              device,
              gamma=1,
              target=None,
              clip_grad=False):
    # Sample the data
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    state = torch.cat(batch.state)
    state_next = torch.cat(batch.next_state)
    action = torch.cat(batch.action)
    reward = torch.cat(batch.reward)).unsqueeze(1)

    # Pass it through the model
    Qs = model(state).gather(1, action)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    if target is not None:
        Qs_next = target(state_next).max(1)[0].detach().unsqueeze(1)
    else:
        Qs_next = model(state_next).max(1)[0].detach().unsqueeze(1)

    J = (Qs_next * gamma) + reward

    # Compute Huber loss
    loss = F.smooth_l1_loss(Qs, J)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    if clip_grad:
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return model, loss


def build_mask(available, m, n):
    mask = np.zeros((m, n), dtype=np.int)
    for a in available:
        mask[a[0], a[1]] = 1
    return mask


class MoveCount():
    """Count moves on a (m,n) board"""
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.count = np.zeros((self.m, self.n))

    def update(self, move):
        self.count[move[0], move[1]] += 1


def wythoff_dqn3(epsilon=0.1,
                 gamma=0.5,
                 learning_rate=1e-6,
                 num_episodes=100,
                 batch_size=20,
                 memory_capacity=100,
                 game='Wythoff10x10',
                 network='DQN_mlp',
                 anneal=False,
                 tensorboard=None,
                 update_every=5,
                 double=False,
                 double_update=10,
                 save=False,
                 save_model=False,
                 monitor=None,
                 return_none=False,
                 debug=False,
                 progress=False,
                 seed=None):
    """Learn to play Wythoff's w/ a DQN and e-greedy random exploration.
    """

    # ------------------------------------------------------------------------
    # Init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logs...
    if tensorboard is not None:
        try:
            os.makedirs(tensorboard)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        writer = SummaryWriter(log_dir=tensorboard)

    if monitor is not None:
        monitored = create_monitored(monitor)

    # Env...
    if tensorboard is not None:
        env = create_env(game, monitor=True)
    else:
        env = create_env(game, monitor=False)
    env.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------------
    # Init
    #
    # Scores
    score = 0
    total_reward = 0

    # Agents, etc
    m, n, board, available = peek(env)
    all_possible_moves = create_all_possible_moves(m, n)

    if network == 'DQN':
        player = DQN(m, n, num_actions=len(all_possible_moves)).to(device)
        target = DQN(m, n, num_actions=len(all_possible_moves)).to(device)
    elif network == 'DQN_mlp':
        player = DQN_mlp(m, n, num_actions=len(all_possible_moves)).to(device)
        target = DQN_mlp(m, n, num_actions=len(all_possible_moves)).to(device)
    else:
        raise ValueError("network must DQN or DQN_mlp")
    if double:
        target.load_state_dict(player.state_dict())
        target.eval()
    else:
        target = None

    if debug:
        print(f"---------------------------------------")
        print("Setting up....")
        print(f">>> Device: {device}")
        print(f">>> Network is {player}")
        print(f">>> Memory capacity {memory_capacity} ({batch_size})")

    memory = ReplayMemory(memory_capacity)
    optimizer = optim.Adam(player.parameters(), learning_rate)
    moves = MoveCount(m, n)

    # ------------------------------------------------------------------------
    for episode in range(1, num_episodes + 1):
        # Re-init
        transitions = []
        state = env.reset()
        x, y, board, available = state
        moves.update((x, y))
        if debug:
            print(f"---------------------------------------")
            print(f">>> NEW GAME ({episode}).")
            print(f">>> Initial position ({x}, {y})")
            print(f">>> Initial moves {available}")
            print(f">>> Cold available {locate_cold_moves(x, y, available)}")
            print(f">>> All cold {locate_all_cold_moves(x, y)}")

        # Anneal epsilon?
        if anneal:
            epsilon_e = epsilon * (1.0 / np.log((episode + np.e)))
        else:
            epsilon_e = epsilon

        # -------------------------------------------------------------------
        # Play a game
        steps = 1
        done = False
        while not done:
            # Convert board to a model(state)
            state_hat = torch.from_numpy(np.array(board).reshape(m, n))
            state_hat = state_hat.unsqueeze(0).unsqueeze(1).float().to(device)

            # Get and filter Qs (off GPU)
            Qs = player(state_hat).cpu().detach().numpy().squeeze()
            mask = build_mask(available, m, n).flatten()
            Qs *= mask

            # Choose a move
            index = np.nonzero(mask)[0].tolist()
            move_i = e_greedy(Qs, epsilon=epsilon_e, index=index, mode='numpy')

            # Re-index move_i to match 'available' index
            move_a = index.index(move_i)
            move = available[move_a]
            moves.update(move)

            # Analyze it...
            if move in locate_cold_moves(x, y, available):
                score += (1 - score) / episode

            # Play it
            state_next, reward, done, _ = env.step(move)
            (x_next, y_next, board_next, available_next) = state_next

            # Save transitions, as tensors to be used at training time
            # (onto GPU)
            total_reward += reward
            transitions.append([
                # S
                state_hat,
                # A
                torch.tensor(move_i).to(device),
                # S'
                torch.from_numpy(np.array(board_next)).reshape(
                    m, n).unsqueeze(0).unsqueeze(1).float().to(device),
                # R
                torch.tensor([reward]).unsqueeze(0).float().to(device),
            ])

            # -
            if debug:
                print(f">>> state_hat size: {state_hat.shape}")
                print(f">>> state_hat: {state_hat}")
                print(f">>> num available: {len(available)}")
                print(f">>> available: {available}")
                print(f">>> Qs (filtered): {Qs[index]}")
                print(f">>> new position: ({x_next}, {y_next})")

            # Shift states
            state = deepcopy(state_next)
            board = deepcopy(board_next)
            available = deepcopy(available_next)
            x = deepcopy(x_next)
            y = deepcopy(y_next)

            steps += 1

        # ----------------------------------------------------------------
        # Learn from the game
        #
        # Find the losers transition and update its reward w/ -reward
        if steps > 2:
            transitions[-2][3] = transitions[-1][3] * -1

        # Update the memories using the transitions from this game
        for i in range(0, len(transitions)):
            memory.push(*transitions[i])

        if debug:
            print(f">>> final transitions: {transitions[-2:]}")

        # Bypass if we don't have enough in memory to learn
        if episode < batch_size:
            continue

        # Learn, samping a batch of transitions from memory
        player, loss = train_dqn(batch_size,
                                 player,
                                 memory,
                                 optimizer,
                                 device,
                                 target=target,
                                 gamma=gamma)

        # Update target net, if in double mode and time is right.
        if double and (episode % double_update == 0):
            target.load_state_dict(player.state_dict())

        # ----------------------------------------------------------------
        # Logs...
        if progress:
            print(f"---")
        if progress or debug:
            print(f">>> episode: {episode}")
        if debug or progress:
            print(f">>> loss {loss}")
            print(f">>> score: {score}")
            print(f">>> epsilon: {epsilon_e}")

        if tensorboard and (int(episode) % update_every) == 0:
            writer.add_scalar('reward', reward, episode)
            writer.add_scalar('epsilon_e', epsilon_e, episode)
            writer.add_scalar('loss', loss, episode)
            writer.add_scalar('steps', steps, episode)
            writer.add_scalar('score', score, episode)

            # Cold ref:
            cold = create_cold_board(m, n)
            plot_wythoff_board(cold,
                               vmin=0,
                               vmax=1,
                               path=tensorboard,
                               name='cold_board.png')
            writer.add_image('cold_positions',
                             torch.from_numpy(
                                 skimage.io.imread(
                                     os.path.join(tensorboard,
                                                  'cold_board.png'))),
                             0,
                             dataformats='HWC')

            # Extract all value boards, and find extrema
            values = torch.zeros((len(all_possible_moves), m, n))
            for i, a in enumerate(all_possible_moves):
                sample_hat = np.asarray(create_board(a[0], a[1], m, n))

                sample_hat = torch.from_numpy(sample_hat)
                sample_hat = sample_hat.unsqueeze(0).unsqueeze(1).float()

                values[i, :, :] = player(sample_hat).detach().float().reshape(
                    m, n)

            mean_values = torch.mean(values, 0)
            max_values, _ = torch.max(values, 0)
            min_values, _ = torch.min(values, 0)

            # Log
            writer.add_scalar('Q_mean', torch.mean(mean_values), episode)
            writer.add_scalar('Q_min', torch.mean(min_values), episode)
            writer.add_scalar('Q_max', torch.mean(max_values), episode)

            # Plot mean
            plot_wythoff_board(mean_values.numpy(),
                               vmin=mean_values.numpy().min(),
                               vmax=mean_values.numpy().max(),
                               path=tensorboard,
                               name='player_mean_values.png')
            writer.add_image('mean player',
                             torch.from_numpy(
                                 skimage.io.imread(
                                     os.path.join(tensorboard,
                                                  'player_mean_values.png'))),
                             0,
                             dataformats='HWC')
            # Plot max
            plot_wythoff_board(max_values.numpy(),
                               vmin=max_values.numpy().min(),
                               vmax=max_values.numpy().max(),
                               path=tensorboard,
                               name='player_max_values.png')
            writer.add_image('max player',
                             torch.from_numpy(
                                 skimage.io.imread(
                                     os.path.join(tensorboard,
                                                  'player_max_values.png'))),
                             0,
                             dataformats='HWC')
            # Plot min
            plot_wythoff_board(min_values.numpy(),
                               vmin=min_values.numpy().min(),
                               vmax=min_values.numpy().max(),
                               path=tensorboard,
                               name='player_min_values.png')
            writer.add_image('min player',
                             torch.from_numpy(
                                 skimage.io.imread(
                                     os.path.join(tensorboard,
                                                  'player_min_values.png'))),
                             0,
                             dataformats='HWC')

            # Plot move count
            plot_wythoff_board(moves.count,
                               vmax=moves.count.max() / 10,
                               vmin=0,
                               path=tensorboard,
                               name='moves.png')
            writer.add_image('moves',
                             torch.from_numpy(
                                 skimage.io.imread(
                                     os.path.join(tensorboard, 'moves.png'))),
                             0,
                             dataformats='HWC')

        if monitor and (int(episode) % update_every) == 0:
            all_variables = locals()
            for k in monitor:
                monitored[k].append(float(all_variables[k]))

    # --------------------------------------------------------------------
    if save_model:
        state = {
            'stumbler_player_dict': player,
        }
        torch.save(state, save + ".pytorch")
    if monitor:
        save_monitored(save, monitored)
    if tensorboard:
        writer.close()

    result = (player), (score / episode, total_reward)
    if return_none:
        result = None

    return result
