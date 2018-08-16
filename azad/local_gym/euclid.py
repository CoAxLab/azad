from azad.local_gym.wythoff import WythoffEnv


def create_moves(x, y):
    """Create all valid moves from (x, y)"""
    a, b = x, y
    moves = []
    for c in range(max(x, y)):
        if min(a, b) == 0:
            if a >= b:
                moves.append((c, b))
            if b >= a:
                moves.append((a, c))
        elif (max(a, b) - c) % min(a, b) == 0:
            if a >= b:
                moves.append((c, b))
            if b >= a:
                moves.append((a, c))

    return list(set(moves))


class EuclidEnv(WythoffEnv):
    """Euclid's game template. 
    
    Note: subclass to use."""

    def __init__(self, m, n, seed=None):
        super().__init__(m, n, seed)

    def _create_moves(self):
        self.moves = create_moves(self.x, self.y)


class Euclid3x3(EuclidEnv):
    """A 3 by 3 Euclid game"""

    def __init__(self):
        EuclidEnv.__init__(self, m=3, n=3)


class Euclid5x5(EuclidEnv):
    """A 5 by 5 Euclid game"""

    def __init__(self):
        EuclidEnv.__init__(self, m=5, n=5)


class Euclid10x10(EuclidEnv):
    """A 10 by 10 Euclid game"""

    def __init__(self):
        EuclidEnv.__init__(self, m=10, n=10)


class Euclid15x15(EuclidEnv):
    """A 15 by 15 Euclid game"""

    def __init__(self):
        EuclidEnv.__init__(self, m=15, n=15)


class Euclid50x50(EuclidEnv):
    """A 50 by 50 Euclid game"""

    def __init__(self):
        EuclidEnv.__init__(self, m=50, n=50)
