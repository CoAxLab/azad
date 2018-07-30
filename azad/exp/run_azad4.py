"""Run azad experiments"""
import fire

from azad.exp.wythoff4 import WythoffStumbler


def wythoff_stumbler1(num_episodes=3,
                      game="Wythoff10x10",
                      gamma=0.98,
                      epsilon=0.1,
                      learning_rate=1e-3,
                      anneal=True,
                      render=False,
                      tensorboard=None,
                      update_every=5,
                      debug=False,
                      seed_value=None):
    model = WythoffStumbler(
        game=game,
        gamma=gamma,
        epsilon=epsilon,
        anneal=anneal,
        learning_rate=learning_rate,
    )
    model.train(
        num_episodes=num_episodes,
        render=render,
        tensorboard=tensorboard,
        update_every=update_every,
        debug=debug,
        seed_value=seed_value)

    return model


if __name__ == "__main__":
    fire.Fire({
        "wythoff_stumbler1": wythoff_stumbler1,
    })
