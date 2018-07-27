"""Run azad experiments"""
import fire

from azad.exp.wythoff3 import WythoffJumpy
from azad.exp.wythoff3 import WythoffJumpyPuppy


def wythoff_agent1(num_episodes=3,
                   game="Wythoff10x10",
                   delta=1.0,
                   gamma=0.98,
                   epsilon=0.1,
                   learning_rate=1e-3,
                   render=False,
                   tensorboard=None,
                   update_every=5,
                   debug=False,
                   seed_value=None):
    model = WythoffJumpy(
        game=game,
        gamma=gamma,
        epsilon=epsilon,
        delta=delta,
        learning_rate=learning_rate,
    )
    model.train(
        num_episodes=num_episodes,
        render=render,
        tensorboard=tensorboard,
        update_every=update_every,
        debug=debug,
        seed_value=seed_value)


def wythoff_agent2(num_episodes=3,
                   game="Wythoff10x10",
                   gamma=0.98,
                   epsilon=0.1,
                   learning_rate_l=0.5,
                   learning_rate_w=0.01,
                   learning_rate=0.01,
                   render=False,
                   tensorboard=None,
                   update_every=5,
                   debug=False,
                   seed_value=None):
    model = WythoffJumpyPuppy(
        game=game,
        gamma=gamma,
        epsilon=epsilon,
        learning_rate=learning_rate,
        learning_rate_l=learning_rate_l,
        learning_rate_w=learning_rate_w,
        seed_value=seed_value)
    model.train(
        num_episodes=num_episodes,
        render=render,
        tensorboard=tensorboard,
        update_every=update_every,
        debug=debug,
        seed_value=seed_value)


if __name__ == "__main__":
    fire.Fire({
        "wythoff_agent1": wythoff_agent1,
        "wythoff_agent2": wythoff_agent2
    })
