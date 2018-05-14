SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/type/Code/azad/data/

# ---------------------------------------------------------------
# 5-10-2018
# Pole cart: Params that follow are the best I've found following a day or so of
# manual hyperparam opt.
exp0:
	python run_azad.py cart_1 $(DATA_PATH)/exp0 --num_episodes=400 --epsilon_max=0.1 --gamma=0.8 --learning_rate=0.001 --num_hidden=256 

# ---------------------------------------------------------------
# 5-14-2018
# Some intial bandit exps. Hyperparams are from some light manual tuning.

# 2 arm, determisitic on arm '0'
exp1:
	python run_azad.py bandit_1 $(DATA_PATH)/exp1 --num_trials=200 --epsilon=0.2 --learning_rate=0.1


# 2 arm, 80/20
exp2: 
	python run_azad.py bandit_1 $(DATA_PATH)/exp2 --bandit_name=BanditTwoArmedHighLowFixed --num_trials=200 --epsilon=0.2 --learning_rate=0.1

# ---------------------------------------------------------------
