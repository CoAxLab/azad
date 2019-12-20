from math import sqrt
from math import log
from random import choice
from copy import deepcopy

import numpy as np


def random_policy(available):
    return choice(available)


def shift_player(player):
    if player == 0:
        return 1
    elif player == 1:
        return 0
    else:
        raise ValueError("player must be 1 or 0")


def rollout(player, state, mcts, env, default_policy):
    """Rollout until the game is done."""

    n_step = 0
    transitions = []
    while True:
        _, _, _, available = state
        action = default_policy(available)
        state_next, reward, done, info = env.step(action)

        transitions.append((player, state, action, state_next, reward))
        state = deepcopy(state_next)

        if done:
            info['n_step'] = n_step
            info['player'] = player
            return transitions, reward, done, info

        player = shift_player(player)
        n_step += 1


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

    def select(self, available):
        """Pick a move."""
        return self.default_policy(available)


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
        n = 1
        for c in node.children:
            n += c.count

        # Est. weights
        w_exploit = node.value / n
        w_explore = sqrt(log(node.count) / n)
        w_total = w_exploit + self.c * w_explore

        return w_total

    def select(self, node, available):
        """Select the best node (UCB). If there are untested nodes, 
        returns None; use expand() instead?
        """

        # Check for expand
        index = []
        for i, a in enumerate(available):
            if a in node.child_names:
                index.append(node.child_names.index(a))

        # There are no selection options available. Return None.
        if len(index) == 0:
            return None

        # Select the best, by UCB (filtered)
        best = max([node.children[i] for i in index],
                   key=self.upper_conf_bound)
        self.path.append(best)
        return best

    def expand(self, node, available):
        """Expand the tree with a random new action, which should be valued
        by a rollout.
        """
        # Find new candidate actions
        new_actions = []
        for a in available:
            if a not in node.child_names:
                new_actions.append(a)

        # Pick one
        action = self.default_policy(new_actions)

        # Add the action as a node to the growing tree....
        new = Node(name=action, initial_count=1, initial_value=0)
        node.add(new)  # inplace update
        self.path.append(new)

        return new

    def backpropagate(self, winner, reward):
        """Backpropagate a reward along the path."""

        # Update winners value
        for i in range(winner, len(self.path), 2):
            self.path[i].value += reward

        # Update all counts
        for p in self.path:
            p.count += 1

    def reset(self):
        """Reset the path, and return the root node."""
        self.path = []
        return self.root


# ----------------------------
# For PerceptronMCTS:
# Choose and init 3-layer Perceptron

# Do policy improvement:
#   Alternate: create rollouts by pi(Perceptron), and ANN training from last rollouts.

# ----------------------------
# For ResNetMCTS, that is 'AlphaZero':
# Choose and init a ResNet

# Do policy improvement:
#   Alternate: create rollouts by pi(ResNet), and ANN training from last rollouts.
