from math import sqrt
from math import log
from random import choice
from copy import deepcopy

import numpy as np


def random_policy(moves):
    return choice(moves)


def shift_player(player):
    if player == 0:
        return 1
    elif player == 1:
        return 0
    else:
        raise ValueError("player must be 1 or 0")


class MoveCount():
    """Count moves on a (m,n) board"""
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.count = np.zeros((self.m, self.n))

    def update(self, move):
        self.count[move[0], move[1]] += 1


class OptimalCount():
    """Count optimal moves."""
    def __init__(self, intial=0):
        self.num_optimal = intial
        self.counts = 1

    def increase(self):
        self.counts += 1
        self.num_optimal += 1

    def decrease(self):
        self.counts += 1

    def score(self):
        return self.num_optimal / self.counts


class HistoryMCTS():
    def __init__(self):
        self.history = {}

    def add(self, name, mcts):
        self.history[name] = mcts

    def get(self, name):
        if name in self.history:
            return self.history[name]
        else:
            return None


class Node(object):
    """I'm a node, for an MCTS tree."""
    def __init__(self, name, initial_value=0, initial_count=1, children=None):
        self.name = name
        self.value = initial_value
        self.count = initial_count

        self.children = []
        self.child_names = []
        if children is not None:
            for child in children:
                self.add(child)

    def add(self, node):
        """Add a new node to this node.

        NOTE: It's name must be unique to be added, otherwise it is
              ignored.
        """

        assert isinstance(node, Node)
        if node.name not in self.child_names:
            self.children.append(node)
            self.child_names.append(node.name)


class Default(object):
    def __init__(self, default_policy=random_policy):
        """Use only the default policy.

        NOTE: no value functions nor trees are learned. 

        Params
        ------
        default_policy : fn
            The action policy for all choices.
        """
        self.default_policy = default_policy

    def select(self, moves):
        """Pick a move."""
        return self.default_policy(moves)


class MCTS(object):
    def __init__(self, c=1.41, default_policy=random_policy):
        """Monte carlo tree search.

        Params
        ------
        c : float, > 0
            The exploration weight
        default_policy : fn
            The action policy for expansion and rollout

        Implemented as described in:
        https://www.aaai.org/Papers/AIIDE/2008/AIIDE08-036.pdf
        """

        self.default_policy = default_policy
        self.path = []
        self.root = Node(None)

        if c <= 0:
            raise ValueError("c must be postive.")
        self.c = c

    def add(self, node, action):
        new = Node(name=action, initial_count=1, initial_value=0)
        node.add(new)  # inplace update
        self.path.append(new)
        return new

    def upper_conf_bound(self, node):
        """Upper confidence bound"""

        # Count future traversals
        N = 1
        for child in node.children:
            N += child.count

        # Est. weights
        w_exploit = node.value / N
        w_explore = sqrt(log(node.count) / N)
        w_total = w_exploit + self.c * w_explore

        return w_total

    def select(self, node):
        """Select the best node (UCB). If there are untested nodes, 
        returns None; use expand() instead?
        """

        best = max(node.children, key=self.upper_conf_bound)
        self.path.append(best)

        return best

    def expand(self, node, available):
        """Expand the tree with a random new action, which should be valued
        by a rollout.
        """
        if len(node.children) > 0:
            raise ValueError("expand called wrongly")

        # Find new candidate actions, and create Nodes for each.
        # This will leave them available for select later.
        for a in available:
            new = Node(name=a, initial_count=1, initial_value=0)
            node.add(new)  # inplace update

        # Pick a move to rollout, and add it to the path.
        move = self.default_policy(available)
        loc = available.index(move)
        self.path.append(node.children[loc])

        return move, node

    def rollout(self, player, env):
        """Rollout until the game is done."""
        scratch = deepcopy(env)

        done = False
        while not done:
            available = scratch.moves
            action = self.default_policy(available)
            state, reward, done, info = scratch.step(action)
            if not done:
                player = shift_player(player)

        info['player'] = player

        return state, reward, done, info

    def backpropagate(self, winner, reward):
        """Backpropagate a reward along the path."""

        # Update winners value
        for i in range(winner, len(self.path), 2):
            self.path[i].value += reward
            if self.path[i].value < 0:
                self.path[i].value = 0

        # Update all counts
        for p in self.path:
            p.count += 1

    def reset(self):
        """Reset the path, and return the root node."""
        self.path = []
        return self.root

    def expanded(self, node):
        return len(node.children) > 0

    def best(self):
        """Find the best move."""
        i = np.argmax([c.count for c in self.root.children])
        return self.root.children[i].name


def run_mcts(player,
             env,
             num_simulations=10,
             c=1.41,
             default_policy=random_policy,
             mcts=None):

    if mcts is None:
        mcts = MCTS(c=c, default_policy=default_policy)

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
            move, node = mcts.expand(node, scratch_env.moves)
            _, reward, done, _ = scratch_env.step(move)
        if not done:
            _, reward, done, info = mcts.rollout(scratch_player, scratch_env)
            scratch_player = info["player"]

        # Learn
        mcts.backpropagate(scratch_player, reward)

    return mcts.best(), mcts
