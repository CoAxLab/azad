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
	python run_azad.py wythoff_stumbler --tensorboard=$(DATA_PATH)/wythoff/exp0 --save=$(DATA_PATH)/wythoff/exp0/exp0 --num_episodes=5 --update_every=100 --learning_rate=0.1 --epsilon=0.5 --gamma=0.98 --game=Wythoff10x10 --debug=False --anneal=True

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
# NOT USEFUL. OUTDATED.
wythoff_exp4:
	-rm -rf $(DATA_PATH)/wythoff/exp4*
	sleep 5  # Wait for tensorboard to notice the deletion
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/exp4.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python run_azad.py wythoff_strategist $(DATA_PATH)/wythoff/exp4/exp4_n{1}_stb{2}_str{3}_ep{4} --num_trials=15000 --num_stumbles={1} --num_evals=1 --stumbler_learning_rate={2} --strategist_learning_rate={3} --epsilon={4} --stumbler_game=Wythoff10x10 --strategist_game=Wythoff50x50 --tensorboard=False --save=True --debug=False' ::: \
		1 100 ::: 0.01 0.1 0.5 ::: 0.1 0.01 ::: 0 0.1

# An SS network that works, somewhat well.
# c33b60cab330cddda6e00f9f85ee07debb525e0b
wythoff_exp5:
	-rm -rf $(DATA_PATH)/wythoff/exp5
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_stumbler_strategist \
		--num_episodes=10000 \
		--num_stumbles=500 \
		--learning_rate_stumbler=0.1 \
		--stumbler_game=Wythoff15x15 \
		--epsilon=0.5 \
		--anneal=True \
		--gamma=0.98 \
		--num_strategies=500 \
		--learning_rate_strategist=0.01 \
		--strategist_game=Wythoff50x50 \
		--cold_threshold=0.0 \
		--hot_threshold=0.5 \
		--hot_value=-1 \
		--cold_value=1 \
		--debug=False \
		--tensorboard=$(DATA_PATH)/wythoff/exp5 \
		--update_every=50 \
		--save=$(DATA_PATH)/wythoff/exp5 \
		--debug=False 

# ----------------------------------------------------------------------------
# Hyperparam searches
# 8-17-2018

# Stumbler
# Result: Some sensitivity to these params. Not a lot. See `exp6_ranked.csv`; 
# lr = 0.4 ep = 0.4, gamma = 0.5 are good middle of the optimal range
# choices.
# 0d61fc38858adcb8ba53da434734d8a9e68917c4
wythoff_exp6:
	-rm -rf $(DATA_PATH)/wythoff/exp6
	-mkdir $(DATA_PATH)/wythoff/exp6
	# Generate a grid,
	python run_azad.py create_grid $(DATA_PATH)/wythoff/exp6/grid.csv \
		--learning_rate='(0.01, 1.0, 10)' \
		--epsilon='(1.0, 0.1, 10)' \
		--gamma='(0.1, 1.0, 10)'
	# and search it.
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp6/exp6.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		'python run_azad.py wythoff_stumbler --save=$(DATA_PATH)/wythoff/exp6/run_{row_code} --num_episodes=50000 --learning_rate={learning_rate} --epsilon={epsilon} --gamma={gamma} --game=Wythoff15x15 --debug=False --anneal=True' :::: \
		$(DATA_PATH)/wythoff/exp6/grid.csv

# Stumbler-strategist v1
# Learning
# (Stumbler params are based on exp6 results)

# Result: Very little, almost no, senstivity to these num ranges. 
# lr > 0.02 and < 0.06 are similiar. Below lr = 0.01 is very high error. Avoid.
# See `exp7_ranked.csv`.
# 0d61fc38858adcb8ba53da434734d8a9e68917c4
wythoff_exp7:
	-rm -rf $(DATA_PATH)/wythoff/exp7
	-mkdir $(DATA_PATH)/wythoff/exp7
	# Generate a grid,
	python run_azad.py create_grid $(DATA_PATH)/wythoff/exp7/grid.csv --fmt='%i,%.6f,%i,%i' \
		--learning_rate_strategist='(0.001, 0.1, 10)' \
		--num_strategies='(100, 1000, 3)' \
		--num_stumbles='(100, 1000, 3)' 
	# and search it.
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp7/exp7.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		'python run_azad.py wythoff_stumbler_strategist --num_episodes=100 --num_stumbles={num_stumbles} --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies={num_strategies} --learning_rate_strategist={learning_rate_strategist} --strategist_game=Wythoff50x50 --cold_threshold=0.0 --hot_threshold=0.5 --hot_value=-1 --cold_value=1 --debug=False --save=$(DATA_PATH)/wythoff/exp7/run_{row_code} --debug=False' :::: \
		$(DATA_PATH)/wythoff/exp7/grid.csv

# Stumbler-strategist v2

# H/C 
# (num_* and learning_rate_* take from exp6 and 7)

# Result: the H/C threshold does not seem to matter. See `exp8_ranked.csv`
# (0/0) will work.... but....
# seems safer/more conservative to move a little past that. 
# Going w/ 0.2/-0.2.
# 0d61fc38858adcb8ba53da434734d8a9e68917c4
wythoff_exp8:
	-rm -rf $(DATA_PATH)/wythoff/exp8
	-mkdir $(DATA_PATH)/wythoff/exp8
	# Generate a grid,
	python run_azad.py create_grid $(DATA_PATH)/wythoff/exp8/grid.csv \
		--hot_threshold='(0.0, 0.99, 10)' \
		--cold_threshold='(0.0, -0.99, 10)' 
	# and search it.
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp8/exp8.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		'python run_azad.py wythoff_stumbler_strategist --num_episodes=100 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold={cold_threshold} --hot_threshold={hot_threshold} --hot_value=-1 --cold_value=1 --debug=False --save=$(DATA_PATH)/wythoff/exp8/run_{row_code} --debug=False' :::: \
		$(DATA_PATH)/wythoff/exp8/grid.csv

# ----------------------------------------------------------------------------
# Sanity check. Test of run of (manually) choosen 'good' hyper-params (exp6-8)

# Stumbler
wythoff_exp9:
	-rm -rf $(DATA_PATH)/wythoff/exp9
	-mkdir $(DATA_PATH)/wythoff/exp9
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_stumbler --tensorboard=$(DATA_PATH)/wythoff/exp9 --save=$(DATA_PATH)/wythoff/exp9/run --monitor='('loss', 'score')' --save_model=True --num_episodes=5 --update_every=1 --learning_rate=0.4 --epsilon=0.4 --gamma=0.5 --game=Wythoff15x15 --debug=False --anneal=True --return_none=True

# Stumbler-strategist
wythoff_exp10:
	-rm -rf $(DATA_PATH)/wythoff/exp10
	-mkdir $(DATA_PATH)/wythoff/exp10
	sleep 5  # Wait for tensorboard to notice the deletion
	python run_azad.py wythoff_stumbler_strategist \
		--num_episodes=100 \
		--num_stumbles=500 \
		--learning_rate_stumbler=0.4 \
		--stumbler_game=Wythoff15x15 \
		--epsilon=0.4 \
		--anneal=True \
		--gamma=0.5 \
		--num_strategies=500 \
		--learning_rate_strategist=0.02 \
		--strategist_game=Wythoff50x50 \
		--cold_threshold=-0.2 \
		--hot_threshold=0.2 \
		--hot_value=-1 \
		--cold_value=1 \
		--debug=False \
		--update_every=50 \
		--save=$(DATA_PATH)/wythoff/exp10/run \
		--save_model=True \
		--debug=False \
		--tensorboard=$(DATA_PATH)/wythoff/exp10 

# wythoff_exp10:
# 	-rm -rf $(DATA_PATH)/wythoff/exp10
# 	-mkdir $(DATA_PATH)/wythoff/exp10
# 	sleep 5  # Wait for tensorboard to notice the deletion
# 	python run_azad.py wythoff_stumbler_strategist \
# 		--num_episodes=100 \
# 		--num_stumbles=500 \
# 		--learning_rate_stumbler=0.4 \
# 		--stumbler_game=Wythoff15x15 \
# 		--epsilon=0.4 \
# 		--anneal=True \
# 		--gamma=0.5 \
# 		--num_strategies=500 \
# 		--learning_rate_strategist=0.02 \
# 		--strategist_game=Wythoff50x50 \
# 		--cold_threshold=-0.2 \
# 		--hot_threshold=0.2 \
# 		--hot_value=-1 \
# 		--cold_value=1 \
# 		--debug=False \
# 		--update_every=10 \
# 		--save=$(DATA_PATH)/wythoff/exp10/run \
# 		--save_model=True \
# 		--stumbler_monitor='('episode', 'loss', 'score')' \
# 		--strategist_monitor='('episode', 'loss')' \
# 		--monitor='('episode', 'influence')' \
# 		--return_none=True \
# 		--debug=False 
		# --tensorboard=$(DATA_PATH)/wythoff/exp10 

# ----------------------------------------------------------------------------
# Exps for paper. Use top N opt params from exp6-7.

# - Stumbler only
# - SS
# - SS w/ only cold
# - SS w/ only cold (no sym sampling)
# - SS w/ only hot
# - SS w/ perfect strategy (+ control)
# - S: Train Wythoff: relearn Nim, Euclid
# - SS: Train Wythoff: relearn Nim, Euclid
# - SS in shallow mode 