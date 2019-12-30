import fire
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import gym

from azad.utils import save_checkpoint
from azad.exp.alternatives.mcts import run_mcts
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
from azad.exp.wythoff import create_monitored


def wythoff_mcts(num_episodes=10,
                 num_simulations=10,
                 c=1.41,
                 game='Wythoff10x10',
                 device="cpu",
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
    num_simulations = int(num_simulations)

    score = 0
    m, n = env.m, env.n

    moves = MoveCount(m, n)
    opts = OptimalCount(0)
    history = HistoryMCTS()
    mcts = None

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

        # Restart vars
        # player = 0
        winner = None
        done = False

        # The root should eventuall be linked to all possible starting
        # configurations.
        if debug:
            print("---")
            print(f">>> New game {n} - ({env.x},{env.y})")
        # --------------------------------------------------------------------
        # Play a game.
        step = 0
        while not done:
            # Use MCTS to choose a move
            mcts = None
            if use_history and ((x, y) in history):
                mcts = history.get((x, y))
                if debug: print(f">>> {step}. using mcts history")

            # mcts = history.get((x, y))
            move, mcts = run_mcts(player,
                                  env,
                                  num_simulations=num_simulations,
                                  c=c,
                                  default_policy=random_policy,
                                  mcts=mcts)

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

            # Shift state for next iterations
            x, y, board, available = state
            player = shift_player(player)

        # --------------------------------------------------------------------
        # Log results
        if monitor and (int(episode) % update_every) == 0:
            all_variables = locals()
            for k in monitor:
                monitored[k].append(float(all_variables[k]))

    # ------------------------------------------------------------------------
    if monitor:
        save_monitored(save, monitored)

    result = dict(mcts=history, score=score)
    if save is not None:
        save_checkpoint(result, filename=save + ".pkl")
    else:
        return result
