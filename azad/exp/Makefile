SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/type/Code/azad/data/

# ---------------------------------------------------------------
# 5-10-2018
# Pole cart: Params that follow are the best I've found following a day or so of
# manual hyperparam opt. After about 100 episodes the pole balanced duration
# should start to hover around 200; 200 is considered winning amount and defines 
# when the problem is 'solved'. 
#
# This isn't a quite perfect tuning. The models hangs around 200 but does not
# consistently peg it. Perfection is possible.
cart_exp0:
	-rm -rf $(DATA_PATH)/cart/exp0
	sleep 3  # Wait for tensorboard to notice the deletion
	python run_azad.py cart_stumbler $(DATA_PATH)/cart/exp0 --num_episodes=4000 --epsilon_max=0.1 --gamma=0.8 --learning_rate=0.001 --num_hidden=256 

# ---------------------------------------------------------------
# 5-14-2018
# Some intial bandit exps. Hyperparams are from some light manual tuning.

# 2 arm, determisitic on arm '0'
bandit_exp0:
	-rm -rf $(DATA_PATH)/bandit/exp0
	sleep 3  # Wait for tensorboard to notice the deletion
	python run_azad.py bandit_stumbler $(DATA_PATH)/bandit/exp0 --num_trials=200 --epsilon=0.2 --learning_rate=0.1

bandit_exp1: 
	-rm -rf $(DATA_PATH)/bandit/exp1
	sleep 3  # Wait for tensorboard to notice the deletion
	python run_azad.py bandit_stumbler $(DATA_PATH)/bandit/exp1 --bandit_name=BanditTwoArmedHighLowFixed --num_trials=200 --epsilon=0.2 --learning_rate=0.1

# ---------------------------------------------------------------
# 5-16-2018
# Testing wythoff 
wythoff_exp0:
	-rm -rf $(DATA_PATH)/wythoff/exp0
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_stumbler $(DATA_PATH)/wythoff/exp0 --num_trials=25000 --learning_rate=0.1 --wythoff_name=Wythoff5x5 --log=True

# Testing wythoff 
wythoff_exp1:
	-rm -rf $(DATA_PATH)/wythoff/exp1
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_strategist $(DATA_PATH)/wythoff/exp1 --num_trials=5000 --num_stumble=100 --learning_rate=0.1 --strategy_default_value=0.5 --strategy_name=estimate_alp_hot_cold --wythoff_name_stumbler='Wythoff5x5' --wythoff_name_strategist='Wythoff10x10' --log=True 

# Make only random choices
wythoff_exp2:
	-rm -rf $(DATA_PATH)/wythoff/exp2
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_random $(DATA_PATH)/wythoff/exp2 --num_trials=25000 --wythoff_name=Wythoff50x50 --log=True


# 6-12-2018
# Simplified testing fn for strategist
# Make only random choices
wythoff_test1:
	-rm -rf $(DATA_PATH)/testing/test1
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py testing_wythoff_strategist ../../data/testing/test1 --strategic_default=0.0 --num_trials=2000 --learning_rate=0.01 --num_hidden1=10 --wythoff_name_stumbler=Wythoff10x10 --wythoff_name_strategist=Wythoff50x50 --log