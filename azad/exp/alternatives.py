import os, csv
import sys

import errno
import pudb

from collections import defaultdict
from copy import deepcopy

import torch
import torch as th
import torch.optim as optim
import torch.nn.functional as F

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

from azad.models import DQN
from azad.models import ReplayMemory
from azad.models import Transition
from azad.policy import epsilon_greedy as e_greedy
from azad.policy import softmax

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


def train_dqn(batch_size, gamma, model, memory, optimizer, device):
    # Sample the data
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    state = torch.cat(batch.state)
    state_next = torch.cat(batch.next_state)
    action = torch.cat(batch.action)
    reward = torch.cat(batch.reward)

    # Pass it through the model
    Qs = model(state).gather(1, action)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    Qs_next = model(state_next).max(1)[0].detach()
    J = (Qs_next * gamma) + reward

    # Compute Huber loss
    loss = F.smooth_l1_loss(Qs, J.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return model, loss


def estimate_Q(available, model):
    Qs = [float(model(torch.array(a))) for a in available]
    return np.asarray(Qs)


def shift_mover(mover):
    if mover == 'player':
        return 'opponent'
    else:
        return 'player'


def analyze_move(score, move, available, episode):
    best = 0.0
    x, y = move
    if cold_move_available(x, y, available):
        if move in locate_cold_moves(x, y, available):
            best = 1.0
            score += (best - score) / (episode + 1)
    return score


def wythoff_dqn(epsilon=0.1,
                gamma=0.8,
                learning_rate=0.1,
                num_episodes=10,
                batch_size=100,
                memory_capacity=10000,
                game='Wythoff10x10',
                model=None,
                opponent=None,
                anneal=False,
                tensorboard=None,
                update_every=5,
                initial=0,
                self_play=False,
                save=False,
                save_model=False,
                monitor=None,
                return_none=False,
                debug=False,
                seed=None):
    """Learn to play Wythoff's w/ e-greedy random exploration.
    
    Note: Learning is based on a player-opponent joint action formalism 
    and tabular Q-learning.
    """

    # ------------------------------------------------------------------------
    # TODO device setup....
    device = None

    # Init env
    if tensorboard is not None:
        try:
            os.makedirs(tensorboard)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        writer = SummaryWriter(log_dir=tensorboard)

    # Create env
    if tensorboard is not None:
        env = create_env(game, monitor=True)
    else:
        env = create_env(game, monitor=False)

    env.seed(seed)
    np.random.seed(seed)

    if monitor is not None:
        monitored = create_monitored(monitor)

    # ------------------------------------------------------------------------
    # Init Agents
    default_Q = 0.0
    m, n, board, available = peek(env)

    player = DQN(in_channels=2, num_actions=board.size)
    opponent = DQN(in_channels=2, num_actions=board.size)

    player_memory = ReplayMemory(memory_capacity)
    opponent_memory = ReplayMemory(memory_capacity)

    player_optimizer = optim.Adam(player.parameters(), learning_rate)
    opponent_optimizer = optim.Adam(opponent.parameters(), learning_rate)

    # ------------------------------------------------------------------------
    for episode in range(initial, initial + num_episodes):
        # Re-init
        x, y, board, available = env.reset()
        board = tuple(flatten_board(board))
        if debug:
            print(f"---------------------------------------")
            print(f">>> NEW GAME ({episode}).")
            print(f">>> Initial position ({x}, {y})")
            print(f">>> Initial moves {available}")
            print(f"---------------------------------------")

        # -------------------------------------------------------------------
        # Anneal epsilon?
        if anneal:
            epsilon_e = epsilon * (1.0 / np.log((episode + np.e)))
        else:
            epsilon_e = epsilon

        # -------------------------------------------------------------------
        # Play a game
        steps = 1
        done = False
        player_win = False
        reward_last = 0
        available_last = deepcopy(available)
        mover = 'opponent'  # This will shift to player on the first move.
        while not done:
            # Choose a mover
            mover = shift_mover(mover)
            if mover == 'player':
                model = player
                memory = opponent_memory  # memory is updated leapfrog
            else:
                model = opponent
                memory = player_memory

            # Choose a move
            Qs = estimate_Q(available, model)
            move_i = e_greedy(Qs, epsilon=epsilon_e, mode='numpy')
            move = available[move_i]

            # Analyze it...
            score = analyze_move(score, move, available, episode)

            # Play it
            state_next, reward, done, _ = env.step(move)
            (x_next, y_next, _, available_next) = state_next
            total_reward += reward
            if debug: print(f">>> {mover}: {move}")

            # Update memory for __last__ players move.
            # But only if have played at least once....
            # If mover wins first play, that gets handled below.
            if steps > 1:
                R = reward_last - reward
                memory.push(available_last, move_last, available, R)

            # Shift
            reward_last = deepcopy(reward)
            available_last = deepcopy(available)
            move_last = deepcopy(move)
            available = deepcopy(available_next)
            state = deepcopy(state_next)

            steps += 1

        # ----------------------------------------------------------------
        # Update winner's memory
        if mover == 'player':
            memory = player_memory
        else:
            memory = opponent_memory
        memory.push(state, move, available, state_next, reward_last)

        # ----------------------------------------------------------------
        # Learn by unrolling the last game...
        player, loss = train_dqn(batch_size, player, player_memory,
                                 player_optimizer, device)
        opponent, loss = train_dqn(batch_size, opponent, opponent_memory,
                                   opponent_optimizer, device)

        # ----------------------------------------------------------------
        if tensorboard and (int(episode) % update_every) == 0:
            writer.add_scalar('reward', reward, episode)
            writer.add_scalar('Q_max', np.max(Qs), episode)
            writer.add_scalar('epsilon_e', epsilon_e, episode)
            writer.add_scalar('stumber_error', loss, episode)
            writer.add_scalar('stumber_steps', steps, episode)
            writer.add_scalar('stumbler_score', score, episode)

            # Cold ref:
            cold = create_cold_board(m, n)
            plot_wythoff_board(
                cold, vmin=0, vmax=1, path=tensorboard, name='cold_board.png')
            writer.add_image(
                'cold_positions',
                skimage.io.imread(os.path.join(tensorboard, 'cold_board.png')))

            # Agent max(Q) boards
            values = expected_value(m, n, player)
            plot_wythoff_board(
                values, path=tensorboard, name='player_max_values.png')
            writer.add_image(
                'player',
                skimage.io.imread(
                    os.path.join(tensorboard, 'player_max_values.png')))

            values = expected_value(m, n, opponent)
            plot_wythoff_board(
                values, path=tensorboard, name='opponent_max_values.png')
            writer.add_image(
                'opponent',
                skimage.io.imread(
                    os.path.join(tensorboard, 'opponent_max_values.png')))

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

    result = (player, opponent), (score, total_reward)
    if return_none:
        result = None

    return result
