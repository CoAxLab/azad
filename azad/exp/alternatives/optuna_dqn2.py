"""Tune the dqn2 model of wythoff's using the opotune lib"""
import optuna
import fire

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from azad.exp.alternatives import wythoff_dqn2
from copy import deepcopy


def _build(trial):
    """Build a nn.Module MLP model"""

    # Sample hidden layers and features
    in_features = 4  # Initial
    n_layers = trial.suggest_int('n_layers', 2, 6)
    layers = []
    for l in range(n_layers):
        out_features = trial.suggest_int(f'{l}', in_features, 20)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = deepcopy(out_features)

    # Output layer topo is fixed
    layers.append(nn.Linear(in_features, 1))

    # Define the nn
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)

    return Model


def _objective(trial):
    """Runs a single HP trial"""

    # Build a new Model
    Model = _build(trial)

    # Sample new HP
    learning_rate = trial.suggest_float("learning_rate", 0.0005, 0.5)
    gamma = trial.suggest_float("gamma", 0.01, 0.5)
    epsilon = trial.suggest_float("epsilon", 0.1, 0.9)

    # Run wythoff_dqn2
    result = wythoff_dqn2(epsilon=epsilon,
                          gamma=gamma,
                          learning_rate=learning_rate,
                          num_episodes=250,
                          batch_size=50,
                          memory_capacity=10000,
                          game=GAME,
                          network=Model,
                          anneal=True,
                          tensorboard=None,
                          update_every=1,
                          double=True,
                          double_update=10,
                          save=False,
                          save_model=False,
                          monitor=None,
                          return_none=False,
                          debug=False,
                          device=DEVICE,
                          clip_grad=True,
                          progress=False,
                          zero=False,
                          seed=SEED)

    return result["score"]  # the final


def optuna_dqn2(save=None,
                num_trials=100,
                game='Wythoff15x15',
                device="cpu",
                debug=True,
                seed=None):
    # Set globals used in _objective
    global DEVICE
    global SEED
    global GAME
    DEVICE = device
    SEED = seed
    GAME = game

    # Run the study
    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=num_trials)
    trial = study.best_trial
    if debug:
        print(f">>> Number of finished trials: {study.trials}")
        print(f">>> Best trial {trial}")
        print(f">>> score: {trial.value}")
        print(f">>> params:\n")
        for k, v in trial.params.items():
            print(f"\t{k}: {v}")

    # Save?
    if save is not None:
        torch.save(study, save)

    return study
