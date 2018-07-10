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
# 5-16-2018
# Testing wythoff stumbler 
wythoff_exp0:
	-rm -rf $(DATA_PATH)/wythoff/exp0
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_stumbler $(DATA_PATH)/wythoff/exp0 --model_name=exp0.pth --num_trials=50000 --learning_rate=0.5 --epsilon=0.2 --gamma=0.98 --game=Wythoff10x10 --tensorboard=True --debug=False

# Testing wythoff strategist 
wythoff_exp1:
	-rm -rf $(DATA_PATH)/wythoff/exp1
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_strategist $(DATA_PATH)/wythoff/exp1 --num_trials=5 --num_stumble=10 --num_evals=1 --stumbler_learning_rate=0.2 --strategist_learning_rate=0.01 --epsilon=0.1 --stumbler_game='Wythoff10x10' --strategist_game='Wythoff50x50' --tensorboard=True --debug=False --save=True

# Strategist learning, directly from the opt. cold board
wythoff_exp2:
	-rm -rf $(DATA_PATH)/wythoff/exp2
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_optimal $(DATA_PATH)/wythoff/exp2 --num_trials=10000 --learning_rate=0.01 --stumbler_game=Wythoff10x10 --strategist_game=Wythoff50x50 --debug=False --tensorboard=True

# ----------------------------------------------------------------------------
# 6-26-2018
# A true Q agent
wythoff_exp3:
	-rm -rf $(DATA_PATH)/wythoff/exp3
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_agent $(DATA_PATH)/wythoff/exp3 --num_trials=50000 --learning_rate=0.25 --epsilon=0.1 --gamma=0.99 --game=Wythoff10x10 --tensorboard=True --debug=True

# ----------------------------------------------------------------------------
# Grid-hyper param search for the wythoff_strategist
wythoff_exp4:
	-rm -rf $(DATA_PATH)/wythoff/exp4*
	sleep 5  # Wait for tensorboard to notice the deletion
	parallel -j 8 -v 
		--joblog '$(DATA_PATH)/osc1000.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python run_azad.py wythoff_strategist $(DATA_PATH)/wythoff/exp4_n{1}_stb{2}_str{3}_ep{4} --num_trials=15000 --num_stumble={1} --num_evals=1 --stumbler_learning_rate={2} --strategist_learning_rate={3} --epsilon={4} --stumbler_game='Wythoff10x10' --strategist_game='Wythoff50x50' --tensorboard=True --debug=False' ::: \
		1 10 100 200 ::: 0.01 0.1 0.2 0.4 ::: 0.1 0.01 0.001 ::: 0 0.1 0.2 0.4