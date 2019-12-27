from copy import deepcopy
from scipy.constants import golden

import numpy as np
import gym


def create_board(i, j, m, n):
    """Create a binary board, with position (i, j) marked."""
    board = np.zeros((m, n))
    board[i, j] = 1.0

    return board


def create_cold_board(m, n, default=0.0, cold_value=1):
    """Create a (m, n) binary board with cold moves as '1'"""
    cold_board = np.ones((m, n)) * default
    for k in range(m - 1):
        mk = int(k * golden)
        nk = int(k * golden**2)
        if (nk < m) and (mk < n):
            cold_board[mk, nk] = cold_value
            cold_board[nk, mk] = cold_value

    return cold_board


def locate_all_cold_moves(m, n):
    """Locate all the cold moves"""
    moves = []
    for k in range(m - 1):
        mk = int(k * golden)
        nk = int(k * golden**2)
        if (nk < m) and (mk < n):
            moves.append((mk, nk))
            moves.append((nk, mk))

    return moves


def cold_move_available(x, y, moves):
    colds = locate_all_cold_moves(x, y)
    for cold in colds:
        if cold in moves:
            return True

    return False


def locate_cold_moves(x, y, moves):
    all_colds = locate_all_cold_moves(x, y)

    colds = []
    for cold in all_colds:
        if cold in moves:
            colds.append(cold)

    return colds


def locate_closest_cold_move(x, y, moves):
    """Locate possible cold moves"""
    cold_moves = locate_all_cold_moves(x, y)
    for cold in cold_moves:
        if cold in moves:
            return cold

    return None


def locate_closest(moves):
    """Find the move closest to the winning position (0,0)."""

    # If the origin is available, take it
    if (0, 0) in moves:
        return (0, 0)

    closest = (np.inf, np.inf)  # HUGE initial value
    for move in moves:
        if (move[0] < closest[0]) or (move[1] < closest[1]):
            closest = deepcopy(move)

    return closest


def create_all_possible_moves(m, n):
    """Create all moves on a (m,n) board."""
    moves = []
    for i in range(m):
        for j in range(n):
            moves.append((i, j))

    return list(set(moves))


def create_moves(x, y):
    """Create all valid moves from (x, y)"""
    if (x == 0) and (y == 0):
        return list([(0, 0)])

    moves = []
    for i in range(x):
        moves.append((i, y))
    for i in range(y):
        moves.append((x, i))

    shortest = min(x, y)
    for i in range(1, shortest + 1):
        moves.append((x - i, y - i))

    return list(set(moves))


def locate_moves(moves, all_possible_moves):
    """Index moves into the total possible set of moves."""
    index = []
    for m in moves:
        try:
            i = all_possible_moves.index(m)
            index.append(i)
        except ValueError:
            pass
    return index


class WythoffEnv(gym.Env):
    """Wythoff's game template. 
    
    Note: subclass to use."""
    def __init__(self, m, n):
        # Board/pile size
        self.m = int(m)
        self.n = int(n)

        self.info = {"m": m, "n": n}

        # Current postition
        self.x = None
        self.y = None
        self.board = None
        self.moves = None

        self.prng = np.random.RandomState(None)

    def seed(self, seed):
        self.prng = np.random.RandomState(seed)

    def step(self, move):
        # Parse the move
        dx, dy = move
        dx = int(dx)
        dy = int(dy)

        # Winning move?
        if (dx == 0) and (dy == 0):
            self.x = dx
            self.y = dy
            reward = 1
            done = True

        # Just move....
        else:
            self.x = dx
            self.y = dy
            reward = 0
            done = False

        # Update state variables
        self._create_board()
        self._create_moves()

        state = (self.x, self.y, self.board, self.moves)

        return state, reward, done, self.info

    def _create_moves(self):
        self.moves = create_moves(self.x, self.y)

    def _reset_board(self):
        self.board = np.zeros((self.m, self.n))

    def _create_board(self):
        self._reset_board()
        self.board[self.x, self.y] = 1

    def reset(self):
        self.x = self.prng.randint(1, self.m)
        self.y = self.prng.randint(1, self.n)

        self._create_board()
        self._create_moves()

        state = (self.x, self.y, self.board, self.moves)

        return state

    def render(self, mode='human', close=False):
        pass


class Wythoff3x3(WythoffEnv):
    """A 3 by 3 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=3, n=3)


class Wythoff5x5(WythoffEnv):
    """A 5 by 5 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=5, n=5)


class Wythoff10x10(WythoffEnv):
    """A 10 by 10 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=10, n=10)


class Wythoff15x15(WythoffEnv):
    """A 15 by 15 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=15, n=15)


class Wythoff50x50(WythoffEnv):
    """A 50 by 50 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=50, n=50)


class Wythoff100x100(WythoffEnv):
    """A 100 by 100 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=100, n=100)


class Wythoff150x150(WythoffEnv):
    """A 150 by 150 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=150, n=150)


class Wythoff200x200(WythoffEnv):
    """A 200 by 200 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=200, n=200)


class Wythoff250x250(WythoffEnv):
    """A 250 by 250 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=250, n=250)


class Wythoff300x300(WythoffEnv):
    """A 300 by 300 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=300, n=300)


class Wythoff350x350(WythoffEnv):
    """A 350 by 350 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=350, n=350)


class Wythoff400x400(WythoffEnv):
    """A 400 by 400 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=400, n=400)


class Wythoff450x450(WythoffEnv):
    """A 450 by 450 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=450, n=450)


class Wythoff500x500(WythoffEnv):
    """A 500 by 500 Wythoff game"""
    def __init__(self):
        WythoffEnv.__init__(self, m=500, n=500)