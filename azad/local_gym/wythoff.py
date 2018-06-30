import numpy as np
import gym


def create_moves(x, y):
    moves = []
    for i in range(x):
        moves.append((i, y))
    for i in range(y):
        moves.append((x, i))

    shortest = min(x, y)
    for i in range(1, shortest + 1):
        moves.append((x - i, y - i))

    return list(set(moves))


class WythoffEnv(gym.Env):
    """Wythoff's game.
    
    Note: the opponent is simulated by a perfect player
    """

    def __init__(self, m, n, seed=None):
        # Board/pile size
        self.m = int(m)
        self.n = int(n)

        self.info = {"m": m, "n": n}

        # Current postition
        self.x = None
        self.y = None
        self.board = None
        self.moves = None

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
        self._place_piece()
        self._create_moves()

        state = (self.x, self.y, self.board, self.moves)

        return state, reward, done, self.info

    def _create_moves(self):
        self.moves = create_moves(self.x, self.y)

    def _reset_board(self):
        self.board = np.zeros((self.m, self.n))

    def _place_piece(self):
        self._reset_board()
        self.board[self.x, self.y] = 1

    def reset(self):
        self.x = self.prng.randint(1, self.m)
        self.y = self.prng.randint(1, self.n)

        self._place_piece()
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