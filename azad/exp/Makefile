SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/type/Code/azad/data/
# DATA_PATH=/home/ejp/src/azad/data/

# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# 5-16-2018 - 8/9/2018 
# Testing wythoff stumbler 
wythoff_exp0:
	-rm -rf $(DATA_PATH)/wythoff/exp0
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_stumbler --tensorboard=$(DATA_PATH)/wythoff/exp0 --num_episodes=50000 --update_every=100 --learning_rate=0.1 --epsilon=5.0 --gamma=0.98 --game=Wythoff10x10 --debug=False --anneal=True

# Testing wythoff strategist 
wythoff_exp1:
	-rm -rf $(DATA_PATH)/wythoff/exp1
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_strategist $(DATA_PATH)/wythoff/exp1 --num_trials=5 --num_stumbles=10 --num_evals=1 --stumbler_learning_rate=0.2 --strategist_learning_rate=0.01 --epsilon=0.1 --stumbler_game=Wythoff10x10 --strategist_game=Wythoff50x50 --tensorboard=False --debug=False --save=True

# Strategist learning, directly from the opt. cold board
wythoff_exp2:
	-rm -rf $(DATA_PATH)/wythoff/exp2
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_optimal $(DATA_PATH)/wythoff/exp2 --num_trials=10000 --learning_rate=0.01 --stumbler_game=Wythoff10x10 --strategist_game=Wythoff50x50 --debug=False --tensorboard=True

# ----------------------------------------------------------------------------
# 7-8-2018
# Grid-hyper param search for the wythoff_strategist
wythoff_exp4:
	-rm -rf $(DATA_PATH)/wythoff/exp4*
	sleep 5  # Wait for tensorboard to notice the deletion
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/exp4.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python run_azad.py wythoff_strategist $(DATA_PATH)/wythoff/exp4/exp4_n{1}_stb{2}_str{3}_ep{4} --num_trials=15000 --num_stumbles={1} --num_evals=1 --stumbler_learning_rate={2} --strategist_learning_rate={3} --epsilon={4} --stumbler_game=Wythoff10x10 --strategist_game=Wythoff50x50 --tensorboard=False --save=True --debug=False' ::: \
		1 100 ::: 0.01 0.1 0.5 ::: 0.1 0.01 ::: 0 0.1

wythoff_exp5:
	-rm -rf $(DATA_PATH)/wythoff/exp5
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_stumbler_strategist \
		--num_episodes=10000 \
		--num_stumbles=250 --learning_rate_stumbler=0.1 --stumbler_game=Wythoff10x10 \
		--epsilon=0.5 --anneal=True --gamma=0.98 \
		--num_strategies=1000 --learning_rate_strategist=0.0001 \
		--strategist_game=Wythoff50x50 \
		--cold_threshold=0.0 --hot_threshold=0.5 \
		--hot_value=-1 --cold_value=1 \
		--debug=False \
		--tensorboard=$(DATA_PATH)/wythoff/exp5 --update_every=50 \
		--save=$(DATA_PATH)/wythoff/exp5.pytorch --debug=False 
