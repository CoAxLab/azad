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
from .wythoff import Wythoff10x10

environments = [
    ['BanditTenArmedRandomFixed', 'v0'],
    ['BanditTenArmedRandomRandom', 'v0'],
    ['BanditTenArmedGaussian', 'v0'],
    ['BanditTenArmedUniformDistributedReward', 'v0'],
    ['BanditTwoArmedDeterministicFixed', 'v0'],
    ['BanditTwoArmedHighHighFixed', 'v0'],
    ['BanditTwoArmedHighLowFixed', 'v0'],
    ['BanditTwoArmedLowLowFixed', 'v0'],
    ['Wythoff3x3', 'v0'],
    ['Wythoff10x10', 'v0'],
]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='azad.local_gym:{}'.format(environment[0]),
        timestep_limit=1,
        nondeterministic=True, )
