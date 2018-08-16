from azad.local_gym.wythoff import WythoffEnv


def create_moves(x, y):
    """Create all valid moves from (x, y)"""
    if (x == 0) and (y == 0):
        return list([(0, 0)])

    moves = []
    for i in range(x):
        moves.append((i, y))
    for i in range(y):
        moves.append((x, i))

    return list(set(moves))


class NimEnv(WythoffEnv):
    """Nim's game template. 
    
    Note: subclass to use."""

    def __init__(self, m, n, seed=None):
        super(NimEnv).__init__(m, n, seed)

    def _create_moves(self):
        self.moves = create_moves(self.x, self.y)


class Nim3x3(NimEnv):
    """A 3 by 3 Nim game"""

    def __init__(self):
        WythoffEnv.__init__(self, m=3, n=3)


class Nim5x5(NimEnv):
    """A 5 by 5 Nim game"""

    def __init__(self):
        WythoffEnv.__init__(self, m=5, n=5)


class Nim10x10(NimEnv):
    """A 10 by 10 Nim game"""

    def __init__(self):
        WythoffEnv.__init__(self, m=10, n=10)


class Nim15x15(NimEnv):
    """A 15 by 15 Nim game"""

    def __init__(self):
        WythoffEnv.__init__(self, m=15, n=15)


class Nim50x50(NimEnv):
    """A 50 by 50 Nim game"""

    def __init__(self):
        WythoffEnv.__init__(self, m=50, n=50)
