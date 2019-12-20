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
from azad.exp.alternatives.mcts import MCTS
from azad.exp.alternatives.mcts import Node
from azad.exp.alternatives.mcts import MoveCount
from azad.exp.alternatives.mcts import OptimalCount
from azad.exp.alternatives.mcts import random_policy
from azad.exp.alternatives.mcts import rollout
from azad.exp.alternatives.mcts import shift_player
from azad.exp.wythoff import save_monitored
from azad.exp.wythoff import create_env
from azad.exp.wythoff import peek
from azad.exp.wythoff import create_all_possible_moves
from azad.local_gym.wythoff import locate_cold_moves
from azad.local_gym.wythoff import locate_all_cold_moves
from azad.exp.wythoff import create_monitored


def wythoff_mcts(num_episodes=10,
                 c=1.41,
                 game='Wythoff10x10',
                 device="cpu",
                 tensorboard=None,
                 update_every=5,
                 monitor=None,
                 save=None,
                 debug=False,
                 progress=False,
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
    score = 0
    m, n, board, available = peek(env)
    all_possible_moves = create_all_possible_moves(m, n)

    # Player model, etc
    mcts = MCTS(c=c)
    moves = MoveCount(m, n)
    opts = OptimalCount(0)
    player = 0
    if debug:
        print(f"---------------------------------------")
        print("Setting up....")
        print(f">>> Device: {device}")
        print(f">>> Network is {mcts}")

    # ------------------------------------------------------------------------
    # Train!
    for n in range(num_episodes):
        # Restart the world
        state = env.reset()
        x, y, board, available = state
        moves.update((x, y))

        # Reset always returns the root node
        node = mcts.reset()

        # Restart vars
        player = 0
        winner = None
        n_step = 0
        transitions = []
        done = False
        do_rollout = False

        # The root should eventuall be linked to all possible starting
        # configurations.
        # node = mcts.add(node, (x, y))
        # print(x, y)
        if debug:
            values = [c.value for c in mcts.root.children]
            counts = [c.count for c in mcts.root.children]
            names = [c.name for c in mcts.root.children]
            print("--------------------")
            print(f">>> NEW GAME: {n}.")
            print(f">>> Initial position ({x}, {y})")
            print(f">>> Initial moves {available}")
            print(f">>> Initial player {player}")
            print(f">>> Cold available {locate_cold_moves(x, y, available)}")
            print(f">>> All cold {locate_all_cold_moves(x, y)}")
            if n > 1:
                print(f">>> MCTS root tree: {names}.")
                print(f">>> MCTS best value: {np.max(values)}.")
                print(
                    f">>> MCTS counts - max: {np.max(counts)}, min: {np.min(counts)}, median: {np.median(counts)}."
                )

        # --------------------------------------------------------------------
        # Play a game.
        while not done:
            # Select?
            next_node = mcts.select(node, available)

            # Or expand?
            if next_node is None:
                next_node = mcts.expand(node, available)
                do_rollout = True

            # Choose the move,
            move = next_node.name

            # and move.
            state_next, reward, done, info = env.step(move)
            (x_next, y_next, board_next, available_next) = state_next

            # Analyze it.
            colds = locate_cold_moves(x, y, available)
            if len(colds) > 0:
                if move in colds:
                    opts.increase()
                else:
                    opts.decrease()
                score = opts.score()

            if debug:
                print(f"---")
                print(f">>> position: ({x}, {y})")
                print(f">>> player: {player}")
                print(f">>> num available: {len(available)}")
                print(f">>> cold moves: {colds}")
                print(f">>> move: ({move})")
                print(f">>> new position: ({x_next}, {y_next})")
                print(f">>> score: {score}")
                print(f">>> rollout: {do_rollout}")

            if done:
                winner = player
                break

            # Either rollout to the end of this game,
            # or continue to play strategically.
            #
            # NOTE: Rollouts do not use the MCTS tree,
            # and are terminal.
            if do_rollout:
                rollout_transitions, reward, done, info = rollout(
                    player, state, mcts, env, random_policy)

                n_step += info['n_step']
                player = info["player"]
                winner = player

                transitions.extend(rollout_transitions)
                if debug:
                    print(f">>> rollout...")
            else:
                transitions.append((player, state, move, state_next, reward))

                # Shift states
                node = next_node
                player = shift_player(player)
                state = deepcopy(state_next)
                board = deepcopy(board_next)
                available = deepcopy(available_next)
                x = deepcopy(x_next)
                y = deepcopy(y_next)
                n_step += 1

        # --------------------------------------------------------------------
        # Learn from the game
        mcts.backpropagate(player, reward)

        if debug or progress:
            print(f">>> last score: {score}")
        if debug:
            print(f">>> learning...")

        # --------------------------------------------------------------------
        # Log results
        if monitor and (int(n) % update_every) == 0:
            all_variables = locals()
            for k in monitor:
                monitored[k].append(float(all_variables[k]))

    # ------------------------------------------------------------------------
    if monitor:
        save_monitored(save, monitored)

    result = dict(mcts=mcts, score=score)
    if save is not None:
        save_checkpoint(result, filename=save)
    else:
        return result
