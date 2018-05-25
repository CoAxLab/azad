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

environments = [['BanditTenArmedRandomFixed', 'v0', 1],
                ['BanditTenArmedRandomRandom', 'v0', 1],
                ['BanditTenArmedGaussian', 'v0', 1],
                ['BanditTenArmedUniformDistributedReward', 'v0', 1],
                ['BanditTwoArmedDeterministicFixed', 'v0', 1],
                ['BanditTwoArmedHighHighFixed', 'v0', 1],
                ['BanditTwoArmedHighLowFixed', 'v0', 1],
                ['BanditTwoArmedLowLowFixed', 'v0', 1],
                ['Wythoff3x3', 'v0', 1000], ['Wythoff10x10', 'v0', 1000],
                ['Wythoff15x15', 'v0', 1000], ['Wythoff50x50', 'v0', 1000]]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='azad.local_gym:{}'.format(environment[0]),
        timestep_limit=environment[2],
        nondeterministic=True, )
