import fire
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import gym

from azad.models import ReplayMemory
from azad.utils import save_checkpoint
from azad.exp.alternatives.alphazero import run_alphazero
from azad.exp.alternatives.alphazero import ResNet
from azad.exp.alternatives.alphazero import AlphaZero
from azad.exp.alternatives.alphazero import train_resnet

from azad.exp.alternatives.mcts import MoveCount
from azad.exp.alternatives.mcts import HistoryMCTS
from azad.exp.alternatives.mcts import OptimalCount
from azad.exp.alternatives.mcts import random_policy
from azad.exp.alternatives.mcts import shift_player

from azad.exp.wythoff import save_monitored
from azad.exp.wythoff import create_env
from azad.exp.wythoff import peek
from azad.exp.wythoff import create_all_possible_moves
from azad.exp.wythoff import create_board

from azad.local_gym.wythoff import locate_cold_moves
from azad.local_gym.wythoff import locate_all_cold_moves
from azad.exp.wythoff import create_monitored


def wythoff_alphazero(num_episodes=10,
                      batch_size=100,
                      c=1.41,
                      game='Wythoff10x10',
                      learning_rate=1e-3,
                      device="cpu",
                      max_size=500,
                      tensorboard=None,
                      update_every=5,
                      monitor=None,
                      use_history=False,
                      save=None,
                      debug=False,
                      seed=None):
    """Learn to play Wythoff's, using MCTS."""

    # ------------------------------------------------------------------------
    # Logs...
    if tensorboard is not None:
        raise NotImplementedError()

    # if tensorboard is not None:
    #     try:
    #         os.makedirs(tensorboard)
    #     except OSError as exception:
    #         if exception.errno != errno.EEXIST:
    #             raise
    #     writer = SummaryWriter(log_dir=tensorboard)

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
    num_episodes = int(num_episodes)
    score = 0

    # Play logs
    m, n = env.m, env.n
    moves = MoveCount(m, n)
    opts = OptimalCount(0)
    history = HistoryMCTS()

    # Network learning
    memory = ReplayMemory(1e3)
    network = ResNet(board_size=max_size).to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    if debug:
        print(f">>> Setting up....")
        print(f">>> Device: {device}")

    # ------------------------------------------------------------------------
    # Train!
    for episode in range(num_episodes):
        # Choose player 0 or 1 to start
        player = int(np.random.binomial(1, 0.5))

        # Restart the world
        state = env.reset()
        x, y, board, available = state
        moves.update((x, y))
        if debug:
            print("---")
            print(f">>> Game {episode} - ({env.x},{env.y})")

        # --------------------------------------------------------------------
        # Play a game.
        done = False
        step = 0
        while not done:
            move, mcts = run_alphazero(player,
                                       env,
                                       network,
                                       c=c,
                                       default_policy=random_policy,
                                       device=device)

            # Play it.
            state, reward, done, info = env.step(move)

            # Analyze it.
            colds = locate_cold_moves(x, y, available)
            if len(colds) > 0:
                if move in colds:
                    opts.increase()
                else:
                    opts.decrease()
                score = opts.score()

            # -
            if debug:
                print(f">>> {step}. player: {player}")
                print(f">>> {step}. moves: {available}")
                print(f">>> {step}. cold moves: {colds}")
                print(f">>> {step}. move: ({move})")
                print(f">>> {step}. score: {score}")

            # Log history (only imporved models get stored in the history)
            history.add((x, y), score, mcts)
            moves.update((x, y))
            step += 1

            # Update the memory
            N = sum([node.count for node in mcts.root.children])
            loc = mcts.root.child_names.index(move)
            node = mcts.root.children[loc]
            memory.push(
                torch.tensor([network.all_moves.index(move)],
                             device=device).unsqueeze(0).int(),
                torch.tensor(create_board(x, y, m, n), device=device).float(),
                torch.tensor([node.count / N], device=device).unsqueeze(0),
                torch.tensor([node.value / node.count],
                             device=device).unsqueeze(0),
            )

            # Shift state for next iterations
            x, y, board, available = state
            player = shift_player(player)

        # --------------------------------------------------------------------
        # train the value resnet
        network, loss = train_resnet(network,
                                     memory,
                                     optimizer,
                                     batch_size=batch_size,
                                     clip_grad=False)
        if debug:
            print(f">>> Traning the resnet. Loss: {loss}")

        # --------------------------------------------------------------------
        # Log results
        if monitor and (int(episode) % update_every) == 0:
            all_variables = locals()
            for k in monitor:
                monitored[k].append(float(all_variables[k]))

    # ------------------------------------------------------------------------
    if monitor:
        save_monitored(save, monitored)

    result = dict(mcts=history, network=network, loss=loss, score=score)
    if save is not None:
        save_checkpoint(result, filename=save + ".pkl")
    else:
        return result
