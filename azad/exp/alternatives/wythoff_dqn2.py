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
                        ('state', 'mask', 'action', 'next_state', 'reward'))


class DQN_mlp(nn.Module):
    """Layers for a Deep Q Network, based on a simple MLP."""
    def __init__(self, num_actions, num_hidden1=1000, num_hidden2=2000):
        super(DQN_mlp, self).__init__()
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2

        self.fc1 = nn.Linear(2, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_hidden2)
        self.fc4 = nn.Linear(num_hidden2, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def train_dqn(batch_size, model, memory, optimizer, device, gamma=1):
    # Sample the data
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    state = torch.cat(batch.state)
    state_next = torch.cat(batch.next_state)
    action = np.vstack(batch.action)
    action = torch.from_numpy(action)
    reward = torch.from_numpy(np.asarray(batch.reward)).unsqueeze(1)

    # Pass it through the model
    Qs = model(state).gather(1, action)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    Qs_next = model(state_next).max(1)[0].detach().unsqueeze(1)
    J = (Qs_next * gamma) + reward

    # Compute Huber loss
    loss = F.smooth_l1_loss(Qs, J)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return model, loss


def shift_mover(mover):
    if mover == 'player':
        return 'opponent'
    else:
        return 'player'


def shift_memory(mover, player_memory, opponent_memory):
    if mover == 'player':
        return opponent_memory
    else:
        return player_memory


def shift_model(mover, player, opponent):
    if mover == 'player':
        return opponent
    else:
        return player


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


def wythoff_dqn2(epsilon=0.1,
                 gamma=0.8,
                 learning_rate=0.1,
                 num_episodes=10,
                 batch_size=100,
                 memory_capacity=10000,
                 game='Wythoff10x10',
                 network='DQN_mlp',
                 anneal=False,
                 tensorboard=None,
                 update_every=5,
                 self_play=False,
                 save=False,
                 save_model=False,
                 monitor=None,
                 return_none=False,
                 debug=False,
                 progress=False,
                 seed=None):
    """Learn to play Wythoff's w/ e-greedy random exploration.
    
    Note: Learning is based on a player-opponent joint action formalism 
    and tabular Q-learning.
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
    if network == 'DQN_mlp':
        player = DQN_mlp(num_actions=len(all_possible_moves))
        opponent = DQN_mlp(num_actions=len(all_possible_moves))
    else:
        raise ValueError("network must be DQN_mlp")

    if debug:
        print(f"---------------------------------------")
        print("Setting up....")
        print(f">>> Network is {player}")
        print(f">>> Memory capacity {memory_capacity} ({batch_size})")

    player_memory = ReplayMemory(memory_capacity)
    opponent_memory = ReplayMemory(memory_capacity)
    player_optimizer = optim.Adam(player.parameters(), learning_rate)
    opponent_optimizer = optim.Adam(opponent.parameters(), learning_rate)

    moves = MoveCount(m, n)

    # Override memory so there is one shared between them
    if self_play:
        player_memory = opponent_memory

    # ------------------------------------------------------------------------
    for episode in range(1, num_episodes + 1):
        # Re-init
        #
        # Scores
        steps = 1
        done = False
        mover = 'opponent'  # This will shift to player on the first move.
        transitions = []

        # Worlds
        state = env.reset()
        x, y, board, available = state
        board = tuple(flatten_board(board))
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
        while not done:
            # Choose a mover
            mover = shift_mover(mover)
            memory = shift_memory(mover, player_memory, opponent_memory)
            model = shift_model(mover, player, opponent)

            # Convert board to a model(state)
            state_hat = torch.tensor([x,y]).unsqueeze(0).float()

            # Get and filter Qs
            Qs = model(state_hat).float().detach()  # torch
            Qs = Qs.numpy().squeeze()

            mask = build_mask(available, m, n).flatten()
            Qs *= mask

            # Choose a move
            index = np.nonzero(mask)[0].tolist()
            move_i = e_greedy(Qs, epsilon=epsilon_e, index=index, mode='numpy')

            # Re-index move_i to match 'available' index
            move_a = index.index(move_i)
            move = available[move_a]

            # Analyze it...
            if move in locate_cold_moves(x, y, available):
                score += (1 - score) / episode

            # Play it
            state_next, reward, done, _ = env.step(move)
            (x_next, y_next, board_next, available_next) = state_next
            total_reward += reward

            # Save transitions, as tensors to be used at training time
            moves.update(move)
            
            state_hat_next = torch.tensor([x_next,y_next]).unsqueeze(0).float()

            transitions.append([
                state_hat.float(),
                torch.from_numpy(mask),
                torch.tensor(move_i),
                state_hat_next.float(),
                torch.tensor([reward]).unsqueeze(0).float()
            ])

            # Shift states
            state = deepcopy(state_next)
            board = deepcopy(board_next)
            available = deepcopy(available_next)
            x = deepcopy(x_next)
            y = deepcopy(y_next)
            steps += 1

            # -
            if debug:
                print(f">>> {mover}: {move}")
                print(f">>> new position: ({x_next}, {y_next})")

        # ----------------------------------------------------------------
        # Learn from the game
        #
        # Find the losers transition and update its reward w/ -reward
        if steps > 2:
            transitions[-2][4] = transitions[-1][4] * -1

        # Update the memories using the transitions from this game
        for i in range(0, len(transitions), 2):
            s, x, a, sn, r = transitions[i]
            player_memory.push(s.to(device), x.to(device), a.to(device),
                               sn.to(device), r.to(device))
        for i in range(1, len(transitions), 2):
            s, x, a, sn, r = transitions[i]
            opponent_memory.push(s.to(device), x.to(device), a.to(device),
                                 sn.to(device), r.to(device))

        # Bypass is we don't have enough in memory to learn
        if episode < batch_size:
            continue

        # Learn, samping batches of transitions from memory
        player, player_loss = train_dqn(batch_size,
                                        player,
                                        player_memory,
                                        player_optimizer,
                                        device,
                                        gamma=gamma)
        opponent, opponent_loss = train_dqn(batch_size,
                                            opponent,
                                            opponent_memory,
                                            opponent_optimizer,
                                            device,
                                            gamma=gamma)

        # ----------------------------------------------------------------
        # Logs...
        if progress:
            print(f"---")
        if progress or debug:
            print(f">>> episode: {episode}")
            print(f">>> winner: {mover}")
        if debug or progress:
            print(f">>> Q: {Qs}")
            print(f">>> max(Q): {Qs.max()}")
            print(f">>> min(Q): {Qs.min()}")
            print(f">>> stdev(Q): {Qs.std()}")
            print(
                f">>> loss (player: {player_loss}, opponent: {opponent_loss})")
            print(f">>> player score: {score}")
            print(f">>> epsilon: {epsilon_e}")

        if tensorboard and (int(episode) % update_every) == 0:
            writer.add_scalar('reward', reward, episode)
            writer.add_scalar('epsilon_e', epsilon_e, episode)
            writer.add_scalar('player_loss', player_loss, episode)
            writer.add_scalar('opponent_loss', opponent_loss, episode)
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
                example = create_board(a[0], a[1], m, n)
                values[i, :, :] = player(state_hat).detach().float().reshape(
                    m, n)
            mean_values = torch.mean(values, 0)
            # max_values, _ = torch.max(values, 0)
            # min_values, _ = torch.min(values, 0)

            # Log
            writer.add_scalar('Q_mean', torch.mean(mean_values), episode)

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
            'stumbler_opponent_dict': opponent
        }
        torch.save(state, save + ".pytorch")
    if monitor:
        save_monitored(save, monitored)
    if tensorboard:
        writer.close()

    result = (player, opponent), (score / episode, total_reward)
    if return_none:
        result = None

    return result
