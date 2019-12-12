import os, csv
import sys

import errno
import pudb

from collections import defaultdict
from copy import deepcopy

import torch
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
    Qs = [float(model(torch.tensor([a])).detach()) for a in available]
    return np.asarray(Qs)


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


def analyze_move(score, move, available, episode):
    best = 0.0
    x, y = move
    if cold_move_available(x, y, available):
        if move in locate_cold_moves(x, y, available):
            best = 1.0
            score += (best - score) / (episode + 1)
    return score


def use_board(state, flatten=False):
    (_, _, board, _) = state
    if flatten:
        board = flatten_board(board)
    return board


def use_last_n_moves(past_moves, n, default_move):
    state = []  # first form (s, .)
    for i in range(n):
        k = n - i
        if k < 0:
            state.extend(default_move)
        else:
            state.extend(past_moves[k])

    return np.asarray(state)


def build_mask(available, m, n):
    mask = np.zeros((m, n), dtype=np.int)
    for a in available:
        mask[a[0], a[1]] = 1
    return mask


def build_index(available, m, n):
    index = np.arange(m * n).reshape(m, n)
    mask = build_mask(available, m, n)
    index *= mask
    return index[np.nonzero(index)]


def wythoff_dqn1(epsilon=0.1,
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
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    all_possible_moves = create_all_possible_moves(m, n)
    player = DQN(in_channels=n * m, num_actions=len(all_possible_moves))
    opponent = DQN(in_channels=n * m, num_actions=len(all_possible_moves))

    player_memory = ReplayMemory(memory_capacity)
    opponent_memory = ReplayMemory(memory_capacity)
    player_optimizer = optim.Adam(player.parameters(), learning_rate)
    opponent_optimizer = optim.Adam(opponent.parameters(), learning_rate)

    # ------------------------------------------------------------------------
    for episode in range(initial, initial + num_episodes):
        # Re-init
        state = env.reset()
        x, y, board, available = state
        board = tuple(flatten_board(board))
        if debug:
            print(f"---------------------------------------")
            print(f">>> NEW GAME ({episode}).")
            print(f">>> Initial position ({x}, {y})")
            print(f">>> Initial moves {available}")

        # -------------------------------------------------------------------
        # Anneal epsilon?
        if anneal:
            epsilon_e = epsilon * (1.0 / np.log((episode + np.e)))
        else:
            epsilon_e = epsilon

        # -------------------------------------------------------------------
        # Play a game
        steps = 1
        score = 0
        total_reward = 0
        reward_last = 0
        done = False
        player_win = False
        available_last = deepcopy(available)
        mover = 'opponent'  # This will shift to player on the first move.
        past_moves = []
        while not done:
            # Choose a mover
            mover = shift_mover(mover)
            memory = shift_memory(mover, player_memory, opponent_memory)
            model = shift_model(mover, player, opponent)

            # Convert board to a model(state)
            state_hat = torch.from_numpy(np.array(board).reshape(m, n))
            state_hat = state_hat.unsqueeze(0).unsqueeze(1).float()

            # Get and filter Qs
            Qs = model(state_hat).float().detach()  # torch
            Qs = Qs.numpy()

            mask = build_mask(available, m, n).flatten()
            Qs *= mask
            Qs = Qs[np.nonzero(Qs)]

            # Choose a move
            move_i = e_greedy(Qs, epsilon=epsilon_e, mode='numpy')
            move = available[move_i]

            # Analyze it...
            if mover == 'player':
                score = analyze_move(score, move, available, episode)

            # Play it
            state_next, reward, done, _ = env.step(move)
            (x_next, y_next, board_next, available_next) = state_next
            total_reward += reward

            # Update memory for __last__ players move.
            # But only if have played at least once....
            # If mover wins first play, that gets handled below.
            if steps > 1:
                R = reward_last - reward
                memory.push(state_hat_last.float().to(device),
                            torch.from_numpy(mask).to(device),
                            torch.tensor(move_i).to(device),
                            state_hat.float().to(device),
                            torch.tensor([R]).unsqueeze(0).float().to(device))

            # Shift
            reward_last = deepcopy(reward)
            state_hat_last = deepcopy(state_hat)

            state = deepcopy(state_next)
            board = deepcopy(board_next)
            available = deepcopy(available_next)

            # Moves this g
            steps += 1

            # -
            if debug:
                print(f">>> {mover}: {move}")
                print(f">>> new position: ({x_next}, {y_next})")
            if done and debug:
                print(f">>> Winner: {mover}.")

        # ----------------------------------------------------------------
        # Update winner's memory. Find the right, memory, and build final
        # state, then push the update.
        memory = shift_memory(mover, player_memory, opponent_memory)

        # Convert board to a model(state)
        state_hat = torch.from_numpy(np.array(board).reshape(m, n))
        state_hat = state_hat.unsqueeze(0).unsqueeze(1).float()

        memory.push(
            state_hat.to(device),
            torch.from_numpy(mask).to(device),
            torch.tensor(move_i).to(device), state_hat.to(device),
            torch.tensor([reward_last]).unsqueeze(0).float().to(device))

        # ----------------------------------------------------------------
        # Bypass is we don't have enough in memory to learn
        if episode < batch_size:
            continue

        # Learn by unrolling the last game...
        player, player_loss = train_dqn(
            batch_size,
            player,
            player_memory,
            player_optimizer,
            device,
            gamma=gamma)
        opponent, opponent_loss = train_dqn(
            batch_size,
            opponent,
            opponent_memory,
            opponent_optimizer,
            device,
            gamma=gamma)

        if debug:
            print(
                f">>> loss (player: {player_loss}, opponent: {opponent_loss})")
            print(f">>> player score: {score}")
            print(f">>> epsilon: {epsilon_e}")

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


# WIll be the last_n model....
# def wythoff_dqn2(epsilon=0.1,
#                  gamma=0.8,
#                  learning_rate=0.1,
#                  num_episodes=10,
#                  batch_size=100,
#                  memory_capacity=10000,
#                  last_n=None,
#                  game='Wythoff10x10',
#                  model=None,
#                  opponent=None,
#                  anneal=False,
#                  tensorboard=None,
#                  update_every=5,
#                  initial=0,
#                  self_play=False,
#                  save=False,
#                  save_model=False,
#                  monitor=None,
#                  return_none=False,
#                  debug=False,
#                  seed=None):
#     """Learn to play Wythoff's w/ e-greedy random exploration.

#     Note: Learning is based on a player-opponent joint action formalism
#     and tabular Q-learning.
#     """

#     # ------------------------------------------------------------------------
#     # if gpu is to be used
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Init env
#     if tensorboard is not None:
#         try:
#             os.makedirs(tensorboard)
#         except OSError as exception:
#             if exception.errno != errno.EEXIST:
#                 raise
#         writer = SummaryWriter(log_dir=tensorboard)

#     # Create env
#     if tensorboard is not None:
#         env = create_env(game, monitor=True)
#     else:
#         env = create_env(game, monitor=False)

#     env.seed(seed)
#     np.random.seed(seed)

#     if monitor is not None:
#         monitored = create_monitored(monitor)

#     # ------------------------------------------------------------------------
#     # Init Agents
#     default_Q = 0.0
#     m, n, board, available = peek(env)

#     if last_n is not None:
#         player = DQN(in_channels=last_n + 1, num_actions=1)
#         opponent = DQN(in_channels=last_n + 1, num_actions=1)
#     else:
#         m, n, _, _ = peek(env)
#         all_possible_moves = create_all_possible_moves(m, n)
#         player = DQN(in_channels=n * m, num_actions=len(all_possible_moves))
#         opponent = DQN(in_channels=n * m, num_actions=len(all_possible_moves))

#     player_memory = ReplayMemory(memory_capacity)
#     opponent_memory = ReplayMemory(memory_capacity)

#     player_optimizer = optim.Adam(player.parameters(), learning_rate)
#     opponent_optimizer = optim.Adam(opponent.parameters(), learning_rate)

#     # ------------------------------------------------------------------------
#     for episode in range(initial, initial + num_episodes):
#         # Re-init
#         state = env.reset()
#         x, y, board, available = state
#         board = tuple(flatten_board(board))
#         if debug:
#             print(f"---------------------------------------")
#             print(f">>> NEW GAME ({episode}).")
#             print(f">>> Initial position ({x}, {y})")
#             print(f">>> Initial moves {available}")

#         # -------------------------------------------------------------------
#         # Anneal epsilon?
#         if anneal:
#             epsilon_e = epsilon * (1.0 / np.log((episode + np.e)))
#         else:
#             epsilon_e = epsilon

#         # -------------------------------------------------------------------
#         # Play a game
#         steps = 1
#         score = 0
#         total_reward = 0
#         reward_last = 0
#         done = False
#         player_win = False
#         available_last = deepcopy(available)
#         mover = 'opponent'  # This will shift to player on the first move.
#         past_moves = []
#         while not done:
#             # Choose a mover
#             mover = shift_mover(mover)
#             memory = shift_memory(mover, player_memory, opponent_memory)
#             model = shift_model(mover, player, opponent)

#             # Build that state representation, state_hat
#             if last_n is not None:
#                 # Build the state
#                 state_hat = use_last_n_moves(past_moves, last_n, (m, n))
#                 state_hat = torch.from_numpy(state_hat).unsqueeze(0)

#                 # Sample the available actions
#                 Qs = []
#                 for a in available:
#                     Qs.append(model(torch.cat(state_hat, a)).detach().numpy())

#             else:
#                 # Build the state
#                 state_hat = use_board(state)
#                 state_hat = torch.from_numpy(state_hat)
#                 state_hat = state_hat.unsqueeze(0).unsqueeze(1).float()
#                 Qs = model(state_hat).detach()
#                 Qs = Qs.numpy()

#                 # Filter the actions
#                 board = np.asarray(board).reshape(m, n)
#                 Qs_a = np.zeros_like(board)
#                 Qs = Qs.reshape(*board.shape)
#                 for a in available:
#                     Qs_a[a[0], a[1]] = Qs[a[0], a[1]]
#                 Qs = Qs_a[np.nonzero(Qs_a)]

#             # Choose a move
#             move_i = e_greedy(
#                 Qs.flatten(),
#                 epsilon=epsilon_e,
#                 mode='numpy',
#             )
#             move = available[move_i]

#             # Analyze it...
#             if mover == 'player':
#                 score = analyze_move(score, move, available, episode)

#             # Play it
#             state_next, reward, done, _ = env.step(move)
#             (x_next, y_next, _, available_next) = state_next
#             total_reward += reward
#             if debug:
#                 print(f">>> {mover}: {move}")
#                 print(f">>> new position: ({x_next}, {y_next})")
#                 if done: print(f">>> Final move.")

#             # Update memory for __last__ players move.
#             # But only if have played at least once....
#             # If mover wins first play, that gets handled below.
#             if steps > 1:
#                 R = reward_last - reward
#                 memory.push(state_hat_last, np.asarray(past_moves[-1]),
#                             state_hat, R)

#             # Shift
#             reward_last = deepcopy(reward)
#             state_hat_last = deepcopy(state_hat)
#             past_moves.append(move)
#             state = deepcopy(state_next)
#             available = deepcopy(available_next)

#             # Moves this game...
#             steps += 1

#         # ----------------------------------------------------------------
#         # Update winner's memory. Find the right, memory, and build final
#         # state, then push the update.
#         memory = shift_memory(mover, player_memory, opponent_memory)
#         if last_n is not None:
#             state_hat = use_last_n_moves(past_moves, last_n, (m, n))
#         else:
#             state_hat = use_board(state)

#         memory.push(
#             torch.from_numpy(state_hat).float().to(device),
#             torch.tensor(move).to(device),
#             torch.from_numpy(state_hat).float().to(device),
#             torch.tensor([reward_last]).unsqueeze(0).float().to(device))

#         # ----------------------------------------------------------------
#         # Bypass is we don't have enough in memory to learn
#         if episode < batch_size:
#             continue

#         # Learn by unrolling the last game...
#         player, player_loss = train_dqn(
#             batch_size,
#             player,
#             player_memory,
#             player_optimizer,
#             device,
#             gamma=gamma)
#         opponent, opponent_loss = train_dqn(
#             batch_size,
#             opponent,
#             opponent_memory,
#             opponent_optimizer,
#             device,
#             gamma=gamma)

#         if debug:
#             print(
#                 f">>> loss (player: {player_loss}, opponent: {opponent_loss})")
#             print(f">>> player score: {score}")
#             print(f">>> epsilon: {epsilon_e}")

#         # ----------------------------------------------------------------
#         if tensorboard and (int(episode) % update_every) == 0:
#             writer.add_scalar('reward', reward, episode)
#             writer.add_scalar('Q_max', np.max(Qs), episode)
#             writer.add_scalar('epsilon_e', epsilon_e, episode)
#             writer.add_scalar('stumber_error', loss, episode)
#             writer.add_scalar('stumber_steps', steps, episode)
#             writer.add_scalar('stumbler_score', score, episode)

#             # Cold ref:
#             cold = create_cold_board(m, n)
#             plot_wythoff_board(
#                 cold, vmin=0, vmax=1, path=tensorboard, name='cold_board.png')
#             writer.add_image(
#                 'cold_positions',
#                 skimage.io.imread(os.path.join(tensorboard, 'cold_board.png')))

#             # Agent max(Q) boards
#             values = expected_value(m, n, player)
#             plot_wythoff_board(
#                 values, path=tensorboard, name='player_max_values.png')
#             writer.add_image(
#                 'player',
#                 skimage.io.imread(
#                     os.path.join(tensorboard, 'player_max_values.png')))

#             values = expected_value(m, n, opponent)
#             plot_wythoff_board(
#                 values, path=tensorboard, name='opponent_max_values.png')
#             writer.add_image(
#                 'opponent',
#                 skimage.io.imread(
#                     os.path.join(tensorboard, 'opponent_max_values.png')))

#         if monitor and (int(episode) % update_every) == 0:
#             all_variables = locals()
#             for k in monitor:
#                 monitored[k].append(float(all_variables[k]))

#     # --------------------------------------------------------------------
#     if save_model:
#         state = {
#             'stumbler_player_dict': player,
#             'stumbler_opponent_dict': opponent
#         }
#         torch.save(state, save + ".pytorch")
#     if monitor:
#         save_monitored(save, monitored)
#     if tensorboard:
#         writer.close()

#     result = (player, opponent), (score, total_reward)
#     if return_none:
#         result = None

#     return result
