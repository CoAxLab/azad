from gym.envs.registration import register

from .bandit import BanditTenArmedRandomFixed
from .bandit import BanditTenArmedRandomRandom
from .bandit import BanditTenArmedGaussian
from .bandit import BanditTenArmedUniformDistributedReward
from .bandit import BanditTwoArmedDeterministicFixed
from .bandit import BanditTwoArmedHighHighFixed
from .bandit import BanditTwoArmedHighLowFixed
from .bandit import BanditTwoArmedLowLowFixed

from .wythoff import Wythoff3x3
from .wythoff import Wythoff5x5
from .wythoff import Wythoff10x10
from .wythoff import Wythoff15x15
from .wythoff import Wythoff50x50

from .nim import Nim3x3
from .nim import Nim5x5
from .nim import Nim10x10
from .nim import Nim15x15
from .nim import Nim50x50

from .euclid import Euclid3x3
from .euclid import Euclid5x5
from .euclid import Euclid10x10
from .euclid import Euclid15x15
from .euclid import Euclid50x50

environments = [['BanditTenArmedRandomFixed', 'v0',
                 1], ['BanditTenArmedRandomRandom', 'v0',
                      1], ['BanditTenArmedGaussian', 'v0', 1],
                ['BanditTenArmedUniformDistributedReward', 'v0',
                 1], ['BanditTwoArmedDeterministicFixed', 'v0',
                      1], ['BanditTwoArmedHighHighFixed', 'v0',
                           1], ['BanditTwoArmedHighLowFixed', 'v0',
                                1], ['BanditTwoArmedLowLowFixed', 'v0',
                                     1], ['Wythoff3x3', 'v0', 1000],
                ['Wythoff5x5', 'v0', 1000], ['Wythoff10x10', 'v0', 1000], [
                    'Wythoff15x15', 'v0', 1000
                ], ['Wythoff50x50', 'v0', 1000], ['Nim3x3', 'v0', 1000], [
                    'Nim5x5', 'v0', 1000
                ], ['Nim10x10', 'v0', 1000], ['Nim15x15', 'v0', 1000], [
                    'Nim50x50', 'v0', 1000
                ], ['Euclid3x3', 'v0', 1000], ['Euclid5x5', 'v0', 1000], [
                    'Euclid10x10', 'v0', 1000
                ], ['Euclid15x15', 'v0', 1000], ['Euclid50x50', 'v0', 1000]]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='azad.local_gym:{}'.format(environment[0]),
        timestep_limit=environment[2],
        nondeterministic=True,
    )
