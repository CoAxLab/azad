SHELL=/bin/bash -O expand_aliases
# DATA_PATH=/Users/type/Code/azad/data/
# DATA_PATH=/home/ejp/src/azad/data/
DATA_PATH=/home/stitch/Code/azad/data/
# DATA_PATH=/Users/qualia/Code/azad/data

# ----------------------------------------------------------------------------
# Grid test
grid_test:
	run_azad.py create_grid $(DATA_PATH)/test_grid.csv --num_gpu=4 \
		--learning_rate='(0.002, 0.0000001, 10)' \
		--epsilon='(0.5, 0.05, 10)' 

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
	run_azad.py cart_stumbler $(DATA_PATH)/cart/exp0 --num_episodes=4000 --epsilon_max=0.1 --gamma=0.8 --learning_rate=0.001 --num_hidden=256 

# ----------------------------------------------------------------------------
# 5-14-2018
# Some intial bandit exps. Hyperparams are from some light manual tuning.

# 2 arm, determisitic on arm '0'
bandit_exp0:
	-rm -rf $(DATA_PATH)/bandit/exp0
	sleep 3  # Wait for tensorboard to notice the deletion
	run_azad.py bandit_stumbler $(DATA_PATH)/bandit/exp0 --num_trials=200 --epsilon=0.2 --learning_rate=0.1

bandit_exp1: 
	-rm -rf $(DATA_PATH)/bandit/exp1
	sleep 3  # Wait for tensorboard to notice the deletion
	run_azad.py bandit_stumbler $(DATA_PATH)/bandit/exp1 --bandit_name=BanditTwoArmedHighLowFixed --num_trials=200 --epsilon=0.2 --learning_rate=0.1

# ----------------------------------------------------------------------------
# 5-16-2018 - 8/9/2018 
# Testing wythoff stumbler 
wythoff_exp0:
	-rm -rf $(DATA_PATH)/wythoff/exp0
	sleep 5  # Wait for tensorboard to notice the deletion
	run_azad.py wythoff_stumbler --tensorboard=$(DATA_PATH)/wythoff/exp0 --save=$(DATA_PATH)/wythoff/exp0/exp0 --num_episodes=5 --update_every=100 --learning_rate=0.1 --epsilon=0.5 --gamma=0.98 --game=Wythoff10x10 --debug=False --anneal=True

# Testing wythoff strategist 
wythoff_exp1:
	-rm -rf $(DATA_PATH)/wythoff/exp1
	sleep 5  # Wait for tensorboard to notice the deletion
	run_azad.py wythoff_strategist $(DATA_PATH)/wythoff/exp1 --num_trials=5 --num_stumbles=10 --num_evals=1 --stumbler_learning_rate=0.2 --strategist_learning_rate=0.01 --epsilon=0.1 --stumbler_game=Wythoff10x10 --strategist_game=Wythoff50x50 --tensorboard=False --debug=False --save=True

# Strategist learning, directly from the opt. cold board
wythoff_exp2:
	-rm -rf $(DATA_PATH)/wythoff/exp2
	sleep 5  # Wait for tensorboard to notice the deletion
	run_azad.py wythoff_optimal $(DATA_PATH)/wythoff/exp2 --num_trials=10000 --learning_rate=0.01 --stumbler_game=Wythoff10x10 --strategist_game=Wythoff50x50 --debug=False --tensorboard=True

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
		'run_azad.py wythoff_strategist $(DATA_PATH)/wythoff/exp4/exp4_n{1}_stb{2}_str{3}_ep{4} --num_trials=15000 --num_stumbles={1} --num_evals=1 --stumbler_learning_rate={2} --strategist_learning_rate={3} --epsilon={4} --stumbler_game=Wythoff10x10 --strategist_game=Wythoff50x50 --tensorboard=False --save=True --debug=False' ::: \
		1 100 ::: 0.01 0.1 0.5 ::: 0.1 0.01 ::: 0 0.1

# An SS network that works, somewhat well.
# c33b60cab330cddda6e00f9f85ee07debb525e0b
wythoff_exp5:
	-rm -rf $(DATA_PATH)/wythoff/exp5
	sleep 5  # Wait for tensorboard to notice the deletion
	run_azad.py wythoff_stumbler_strategist \
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
#
# See `notebooks/wythoff_exp6.ipynb` 
wythoff_exp6:
	-rm -rf $(DATA_PATH)/wythoff/exp6
	-mkdir $(DATA_PATH)/wythoff/exp6
	# Generate a grid,
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp6/grid.csv \
		--learning_rate='(0.01, 1.0, 10)' \
		--epsilon='(1.0, 0.1, 10)' \
		--gamma='(0.1, 1.0, 10)'
	# and search it.
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp6/exp6.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		'run_azad.py wythoff_stumbler --save=$(DATA_PATH)/wythoff/exp6/run_{row_code} --num_episodes=50000 --learning_rate={learning_rate} --epsilon={epsilon} --gamma={gamma} --game=Wythoff15x15 --debug=False --anneal=True' :::: \
		$(DATA_PATH)/wythoff/exp6/grid.csv

# Stumbler-strategist v1
# Learning
# (Stumbler params are based on exp6 results)

# Result: Very little, almost no, senstivity to these num ranges. 
# lr > 0.02 and < 0.06 are similiar. Below lr = 0.01 is very high error. Avoid.
#
# See `exp7_ranked.csv`.
# See `notebooks/wythoff_exp7.ipynb` 
#
# 0d61fc38858adcb8ba53da434734d8a9e68917c4
wythoff_exp7:
	-rm -rf $(DATA_PATH)/wythoff/exp7
	-mkdir $(DATA_PATH)/wythoff/exp7
	# Generate a grid,
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp7/grid.csv --fmt='%i,%.6f,%i,%i' \
		--learning_rate_strategist='(0.001, 0.1, 10)' \
		--num_strategies='(100, 1000, 3)' \
		--num_stumbles='(100, 1000, 3)' 
	# and search it.
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp7/exp7.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		'run_azad.py wythoff_stumbler_strategist --num_episodes=100 --num_stumbles={num_stumbles} --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies={num_strategies} --learning_rate_strategist={learning_rate_strategist} --strategist_game=Wythoff50x50 --cold_threshold=0.0 --hot_threshold=0.5 --hot_value=-1 --cold_value=1 --debug=False --save=$(DATA_PATH)/wythoff/exp7/run_{row_code} --debug=False' :::: \
		$(DATA_PATH)/wythoff/exp7/grid.csv

# Stumbler-strategist v2

# H/C 
# (num_* and learning_rate_* take from exp6 and 7)

# Result: the H/C threshold does not seem to matter. See `exp8_ranked.csv`
# (0/0) will work.... but....
#
# seems safer/more conservative to move a little past that. 
# Going w/ 0.2/-0.2.
# 0d61fc38858adcb8ba53da434734d8a9e68917c4
#
# See `notebooks/wythoff_exp8.ipynb` 
wythoff_exp8:
	-rm -rf $(DATA_PATH)/wythoff/exp8
	-mkdir $(DATA_PATH)/wythoff/exp8
	# Generate a grid,
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp8/grid.csv \
		--hot_threshold='(0.0, 0.99, 10)' \
		--cold_threshold='(0.0, -0.99, 10)' 
	# and search it.
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp8/exp8.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		'run_azad.py wythoff_stumbler_strategist --num_episodes=100 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold={cold_threshold} --hot_threshold={hot_threshold} --hot_value=-1 --cold_value=1 --debug=False --save=$(DATA_PATH)/wythoff/exp8/run_{row_code} --debug=False' :::: \
		$(DATA_PATH)/wythoff/exp8/grid.csv

# ----------------------------------------------------------------------------
# Sanity check. Test of run of (manually) choosen 'good' hyper-params (exp6-8)
# Result: both look solid, though opt performance hovers just shy of 90%
# and it should really get to 100?
# Stumbler
wythoff_exp9:
	-rm -rf $(DATA_PATH)/wythoff/exp9
	-mkdir $(DATA_PATH)/wythoff/exp9
	sleep 5  # Wait for tensorboard to notice the deletion
	run_azad.py wythoff_stumbler --tensorboard=$(DATA_PATH)/wythoff/exp9 --save=$(DATA_PATH)/wythoff/exp9/run --monitor='('loss', 'score')' --save_model=True --num_episodes=5 --update_every=1 --learning_rate=0.4 --epsilon=0.4 --gamma=0.5 --game=Wythoff15x15 --debug=False --anneal=True --return_none=True

# Stumbler-strategist
wythoff_exp10:
	-rm -rf $(DATA_PATH)/wythoff/exp10
	-mkdir $(DATA_PATH)/wythoff/exp10
	sleep 5  # Wait for tensorboard to notice the deletion
	run_azad.py wythoff_stumbler_strategist \
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

# Old version, testing various monitor*
# wythoff_exp10:
# 	-rm -rf $(DATA_PATH)/wythoff/exp10
# 	-mkdir $(DATA_PATH)/wythoff/exp10
# 	sleep 5  # Wait for tensorboard to notice the deletion
# 	run_azad.py wythoff_stumbler_strategist \
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

# 8-18-2018
# Adding and searching a `learning_rate_influence` to the SS net
# other params take from exp6-8
# 9e91f8157cac59ae4e3b49933dd0d23ae600dcc4
#
# Result: `learning_rate_influence` is very slack. 
# Anything less than 0.5 is interchangeable. Above that stumbler score 
# variance MAY increase a bit (low n. can't be certain).
# See `exp11_ranked.csv`.
# See `notebooks/wythoff_exp11.ipynb` 
wythoff_exp11:
	-rm -rf $(DATA_PATH)/wythoff/exp11
	-mkdir $(DATA_PATH)/wythoff/exp11
	# Generate a grid,
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp11/grid.csv \
		--learning_rate_influence='(0.01, 1.0, 20)' 
	# and search it.
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp11/exp11.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		'run_azad.py wythoff_stumbler_strategist --num_episodes=100 --learning_rate_influence={learning_rate_influence} --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save=$(DATA_PATH)/wythoff/exp11/run_{row_code} --save_model=True --return_none=True --debug=False' :::: \
		$(DATA_PATH)/wythoff/exp11/grid.csv

# 8-18-2018
# Exploring layer sizes in the strategist, and it's influence
# 821aa24b0af90d4472d5042ed1458cf3286f5d5d
#
# Result: shallow strategists greatly reduce _stumbler_ scores. 
# Any `learning_rate_influence` above 0.2 is prefered.
# Layer size doesn't matter
# See `exp12_ranked.csv`.
# See `notebooks/wythoff_exp12.ipynb` 
wythoff_exp12:
	-rm -rf $(DATA_PATH)/wythoff/exp12
	-mkdir $(DATA_PATH)/wythoff/exp12
	# Generate a grid,
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp12/grid.csv \
		--learning_rate_influence='(0.01, 1.0, 5)' \
		--num_hidden1='(15, 500, 10)' \
		--num_hidden2='(0, 50, 10)' \
	# and search it.
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp12/exp12.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		'run_azad.py wythoff_stumbler_strategist --num_episodes=10 --num_hidden1={num_hidden1} --num_hidden2={num_hidden2} --learning_rate_influence={learning_rate_influence} --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save=$(DATA_PATH)/wythoff/exp12/run_{row_code} --save_model=True --return_none=True --debug=False' :::: \
		$(DATA_PATH)/wythoff/exp12/grid.csv
	
# ----------------------------------------------------------------------------
# Exps for paper. Use hand picked params, based on hyperparam opt. 
# There is lot of slack in many params. I manually choose middle values 
# in slack cases hoping to end up with a good robust set. 
# Parameter sensitivity testing will check these choices later on.

# --- Main effect.
# Stumbler 
wythoff_exp13:
	-rm -rf $(DATA_PATH)/wythoff/exp13
	-mkdir $(DATA_PATH)/wythoff/exp13
	sleep 5  # Wait for tensorboard to notice the deletion
		# and search it.
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp13/exp13.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler --save=$(DATA_PATH)/wythoff/exp13/run_{1} --monitor='('episode', 'loss', 'score', 'total_reward')' --num_episodes=75000 --update_every=10 --learning_rate=0.4 --epsilon=0.4 --gamma=0.5 --game=Wythoff15x15 --debug=False --anneal=True --return_none=True --save_model=True --seed={1}" ::: \
		{1..20}

# SS
# Result: compared to exp13, the SS strongly accelerated learning of optimal play
# and strongly reduced the variance during learning (stabilized learning).
# 211f18c154ae3cac12af74f82d75aceb22f71b92
#
# See `notebooks/wythoff_exp13_14.ipynb`
wythoff_exp14:
	-rm -rf $(DATA_PATH)/wythoff/exp14
	-mkdir $(DATA_PATH)/wythoff/exp14
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp14/exp14.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp14/run_{1} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}


# --- Heuristic controls
# SS w/ only cold
wythoff_exp15:
	-rm -rf $(DATA_PATH)/wythoff/exp15
	-mkdir $(DATA_PATH)/wythoff/exp15
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp15/exp15.parallel.log' \
		--nice 19 --delay 2  \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp15/run_{1} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=None --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# SS w/ only hot
wythoff_exp16:
	-rm -rf $(DATA_PATH)/wythoff/exp16
	-mkdir $(DATA_PATH)/wythoff/exp16
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp16/exp16.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp16/run_{1} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=None --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# SS w/ no sym cold sampling
# 211f18c154ae3cac12af74f82d75aceb22f71b92
#
# Result: HC- and HC are the same. Both offer better performance the H or C
#
# See `notebooks/wythoff_exp15_18.ipynb`
wythoff_exp17:
	-rm -rf $(DATA_PATH)/wythoff/exp17
	-mkdir $(DATA_PATH)/wythoff/exp17
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp17/exp17.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp17/run_{1} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --reflect_cold=False --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# - SS w/ perfect a strategist all the time 
# (a positive control/no strategist learning)
# 211f18c154ae3cac12af74f82d75aceb22f71b92
#
# Result: the perfect strategist offers only a small improvement in performance
# compared to the emperical version
wythoff_exp18:
	-rm -rf $(DATA_PATH)/wythoff/exp18
	-mkdir $(DATA_PATH)/wythoff/exp18
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp18/exp18.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp18/run_{1} --optimal_strategist=True --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# --- Sample top 20 hyperparameters.
# In exp13-17 I explored random seeds w/ fixed hyperparameters.
# Here I fix the seed but explore the top N hyperparameters,
# taken from the joint experiments exp6-12.
#
# See `notebooks/wythoff_joint_param.ipynb` 
#
# Available params:
# cold_threshold,epsilon,gamma,hot_threshold,learning_rate_influence,
# learning_rate_strategist,learning_rate_stumbler,num_hidden1,
# num_hidden2,num_strategies,num_stumbles

# Stumbler
# Result: top_20 params are bad. Stumbler and strategist provide
# equal performance here. Aggregate hyper-params are the choise.

# sets used in exp13-18. 
# 211f18c154ae3cac12af74f82d75aceb22f71b92
#
# See `notebooks/wythoff_exp19_20.ipynb`
wythoff_exp19:
	-rm -rf $(DATA_PATH)/wythoff/exp19
	-mkdir $(DATA_PATH)/wythoff/exp19
	sleep 5  
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp19/exp19.parallel.log' \
		--nice 19 --delay 2  --header : --colsep ',' \
		"run_azad.py wythoff_stumbler --save=$(DATA_PATH)/wythoff/exp19/run_{row_code} --monitor='('episode', 'loss', 'score', 'total_reward')' --num_episodes=75000 --update_every=10 --learning_rate={learning_rate_stumbler} --epsilon={epsilon} --gamma={gamma} --game=Wythoff15x15 --debug=False --anneal=True --return_none=True --save_model=True --seed=42" :::: \
		$(DATA_PATH)/wythoff/joint_ranked.csv

# SS
wythoff_exp20:
	-rm -rf $(DATA_PATH)/wythoff/exp20
	-mkdir $(DATA_PATH)/wythoff/exp20
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp20/exp20.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp20/run_{row_code} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence={learning_rate_influence} --num_stumbles=550 --learning_rate_stumbler={learning_rate_stumbler} --stumbler_game=Wythoff15x15 --epsilon={epsilon} --anneal=True --gamma={gamma} --num_strategies={num_strategies} --learning_rate_strategist={learning_rate_strategist} --strategist_game=Wythoff50x50 --cold_threshold={cold_threshold} --hot_threshold={hot_threshold} --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed=42" :::: \
		$(DATA_PATH)/wythoff/joint_ranked.csv

# --- Self-play
# Result: I thought an older implementation w/ self-play went to 100%
# optimal play. Here that does not hold up. Beyod the params below (from exp13)
# I played w/ several variations. None helped. 
# What has changed? Why no 1.0 score anymore? Epsilon annealing?
# 8771a408d4fc51ca3de658d110e4c434c733c79e 
#
# See `notebooks/wythoff_exp21.ipynb`
wythoff_exp21:
	-rm -rf $(DATA_PATH)/wythoff/exp21
	-mkdir $(DATA_PATH)/wythoff/exp21
	sleep 5  # Wait for tensorboard to notice the deletion
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp21/exp21.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler --save=$(DATA_PATH)/wythoff/exp21/run_{1} --self_play=True --monitor='('episode', 'loss', 'score', 'total_reward')' --num_episodes=75000 --update_every=10 --learning_rate=0.4 --epsilon=0.4 --gamma=0.5 --game=Wythoff15x15 --debug=False --anneal=True --return_none=True --save_model=True --seed={1}" ::: \
		{1..20}

# ----------------------------------------------------------------------------
# Transfer exps for paper

# --- Larger boards (up to 500).
# Greedy stumbler
wythoff_exp24a:
	-rm -rf $(DATA_PATH)/wythoff/exp24a
	-mkdir $(DATA_PATH)/wythoff/exp24a
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp24a/exp24a.aparallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_wythoff --save=$(DATA_PATH)/wythoff/exp24a/run_{1}_{2}.csv --load_model=$(DATA_PATH)/wythoff/exp14/run_{1}.pytorch --num_episodes=1000 --strategist_game={2} --stumbler_game=Wythoff15x15 --return_none=True" ::: \
		{1..20} ::: Wythoff5x5 Wythoff10x10 Wythoff15x15 Wythoff50x50 Wythoff100x100 Wythoff150x150 Wythoff200x200 Wythoff250x250 Wythoff300x300 Wythoff350x350 Wythoff400x400 Wythoff450x450 Wythoff500x500

# Random stumbler
wythoff_exp24b:
	-rm -rf $(DATA_PATH)/wythoff/exp24b
	-mkdir $(DATA_PATH)/wythoff/exp24b
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp24b/exp24b.aparallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_wythoff --save=$(DATA_PATH)/wythoff/exp24b/run_{1}_{2}.csv --load_model=$(DATA_PATH)/wythoff/exp14/run_{1}.pytorch --random_stumbler=True --num_episodes=1000 --strategist_game={2} --stumbler_game=Wythoff15x15 --return_none=True" ::: \
		{1..20} ::: Wythoff5x5 Wythoff10x10 Wythoff15x15 Wythoff50x50 Wythoff100x100 Wythoff150x150 Wythoff200x200 Wythoff250x250 Wythoff300x300 Wythoff350x350 Wythoff400x400 Wythoff450x450 Wythoff500x500


# --- Play new games
# 94dee42aa7f0ba985e7d0086f251f81867d287a2 
# - SS: 
# 1. Train Nim or Euclid
# Nim
wythoff_exp25:
	-rm -rf $(DATA_PATH)/wythoff/exp25
	-mkdir $(DATA_PATH)/wythoff/exp25
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp25/exp25.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp25/run_{1} --monitor='('episode', 'influence', 'eval_score_a', 'eval_score_b')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Nim15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Nim50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# Euclid
wythoff_exp26:
	-rm -rf $(DATA_PATH)/wythoff/exp26
	-mkdir $(DATA_PATH)/wythoff/exp26
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp26/exp26.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp26/run_{1} --monitor='('episode', 'influence', 'eval_score_a', 'eval_score_b')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Euclid15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Euclid50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# 2. Pre-train Wythoff (exp14): Train Nim, Euclid
# Nim
wythoff_exp27:
	-rm -rf $(DATA_PATH)/wythoff/exp27
	-mkdir $(DATA_PATH)/wythoff/exp27
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp27/exp27.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp27/run_{1} --load_model=$(DATA_PATH)/wythoff/exp14/run_{1}.pytorch --new_rules=True --monitor='('episode', 'influence', 'eval_score_a', 'eval_score_b')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Nim15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Nim50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# Euclid
wythoff_exp28:
	-rm -rf $(DATA_PATH)/wythoff/exp28
	-mkdir $(DATA_PATH)/wythoff/exp28
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp28/exp28.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp28/run_{1} --load_model=$(DATA_PATH)/wythoff/exp14/run_{1}.pytorch --new_rules=True --monitor='('episode', 'influence', 'eval_score_a', 'eval_score_b')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Nim15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Nim50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}


# ----------------------------------------------------------------------------
# Parameter senstivity testing (based on exp14)
# 4b8e9c2d41d96f7d96c35b2949eb1cec2b7eca70

# learning_rates (all three)
# influence
wythoff_exp30:
	-rm -rf $(DATA_PATH)/wythoff/exp30
	-mkdir $(DATA_PATH)/wythoff/exp30
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp30/grid.csv \
		--learning_rate_influence='(0.01, 0.4, 20)' 
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp30/exp30.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp30/run_{row_code} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence={learning_rate_influence} --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" :::: \
		$(DATA_PATH)/wythoff/exp30/grid.csv

# stumbler
wythoff_exp31:
	-rm -rf $(DATA_PATH)/wythoff/exp31
	-mkdir $(DATA_PATH)/wythoff/exp31
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp31/grid.csv \
		--learning_rate_stumbler='(0.2, 0.6, 20)' 
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp31/exp31.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp31/run_{row_code} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler={learning_rate_stumbler} --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" :::: \
		$(DATA_PATH)/wythoff/exp31/grid.csv

# strategist
wythoff_exp32:
	-rm -rf $(DATA_PATH)/wythoff/exp32
	-mkdir $(DATA_PATH)/wythoff/exp32
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp32/grid.csv \
		--learning_rate_strategist='(0.01, 0.05, 20)' 
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp32/exp32.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp32/run_{row_code} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist={learning_rate_strategist} --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" :::: \
		$(DATA_PATH)/wythoff/exp32/grid.csv

# cold_threshold/hot_threshold
wythoff_exp33:
	-rm -rf $(DATA_PATH)/wythoff/exp33
	-mkdir $(DATA_PATH)/wythoff/exp33
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp33/grid.csv \
		--hot_threshold='(0.0, 0.4, 10)' 
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp33/exp33.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp33/run_{row_code} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold={hot_threshold} --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" :::: \
		$(DATA_PATH)/wythoff/exp33/grid.csv


wythoff_exp34:
	-rm -rf $(DATA_PATH)/wythoff/exp34
	-mkdir $(DATA_PATH)/wythoff/exp34
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp34/grid.csv \
		--cold_threshold='(0.0, -0.4, 10)'
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp34/exp34.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp34/run_{row_code} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold={cold_threshold} --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" :::: \
		$(DATA_PATH)/wythoff/exp34/grid.csv

# epsilon
wythoff_exp35:
	-rm -rf $(DATA_PATH)/wythoff/exp35
	-mkdir $(DATA_PATH)/wythoff/exp35
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp35/grid.csv \
		--epsilon='(0.01, 0.8, 20)' 
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp35/exp35.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp35/run_{row_code} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon={epsilon} --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" :::: \
		$(DATA_PATH)/wythoff/exp35/grid.csv

# ----------------------------------------------------------------------------
# AAAI reviewer response runs:
# Added a way to turn off the hot/cold part of the strategist. Does learning
# just expected values in the strategist help?
wythoff_exp36:
	-rm -rf $(DATA_PATH)/wythoff/exp36
	-mkdir $(DATA_PATH)/wythoff/exp36
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp36/exp36.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp36/run_{1} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1} --heuristic=False" ::: \
		{1..20}
	
# Try exp37 on larger boards (up to 500).
# (against a Greedy stumbler)
wythoff_exp37:
	-rm -rf $(DATA_PATH)/wythoff/exp37
	-mkdir $(DATA_PATH)/wythoff/exp37
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp37/exp37.aparallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_wythoff --save=$(DATA_PATH)/wythoff/exp37/run_{1}_{2}.csv --load_model=$(DATA_PATH)/wythoff/exp36/run_{1}.pytorch --num_episodes=1000 --strategist_game={2} --stumbler_game=Wythoff15x15 --return_none=True" ::: \
		{1..20} ::: Wythoff5x5 Wythoff10x10 Wythoff15x15 Wythoff50x50 Wythoff100x100 Wythoff150x150 Wythoff200x200 Wythoff250x250 Wythoff300x300 Wythoff350x350 Wythoff400x400 Wythoff450x450 Wythoff500x500

# ----------------------------------------------------------------------------
# NBDT reviewer response runs:
# Note: exp36 addressed one comment.

# R2: Increase stumbler learning rate to 1.0 so stumber updates same 'force' 
# (my words) as the full ss. They think the env is deteriminsitic, which is not
# right because the opponent is, in a meaningful way, part of the env. The
# opponent is not deterministic.
wythoff_exp38:
	-rm -rf $(DATA_PATH)/wythoff/exp38
	-mkdir $(DATA_PATH)/wythoff/exp38
	sleep 5  # Wait for tensorboard to notice the deletion
		# and search it.
	parallel -j 20 -v \
		--joblog '$(DATA_PATH)/wythoff/exp38/exp38.parallel.log' \
		--nice 19 --delay 2 \
		"run_azad.py wythoff_stumbler --save=$(DATA_PATH)/wythoff/exp38/run_{1} --monitor='('episode', 'loss', 'score', 'total_reward')' --num_episodes=75000 --update_every=10 --learning_rate=1.0 --epsilon=0.4 --gamma=0.5 --game=Wythoff15x15 --debug=False --anneal=True --return_none=True --save_model=True --seed={1}" ::: \
		{1..20}

# ----------------------------------------------------------------------------
# 12-13-2019
# 8110924218ee9dbe0035613bb7ba992245678484
#
# DQN testing 
wythoff_exp39:
	-rm -rf $(DATA_PATH)/wythoff/exp39
	-mkdir $(DATA_PATH)/wythoff/exp39
	sleep 5  # Wait for tensorboard to notice the deletion
	run_azad.py wythoff_dqn1 \
		--num_episodes=5000 \
		--batch_size=100 \
		--memory_capacity=1000 \
		--learning_rate=1e-3 \
		--game=Wythoff15x15 \
		--epsilon=0.4 \
		--anneal=False \
		--gamma=0.5 \
		--debug=False \
		--update_every=50 \
		--save=$(DATA_PATH)/wythoff/exp39/run \
		--save_model=True \
		--debug=False \
		--tensorboard=$(DATA_PATH)/wythoff/exp39

# ----------------------------------------------------------------------------
# 12-19-2019
# 38ad00ee33c9acd4ffe3e74eb64ef4af250fc15c
# Tune dqn3 (board representation, self-play). 
# Search learing rate and exploration
wythoff_exp40:
	-rm -rf $(DATA_PATH)/wythoff/exp40
	-mkdir $(DATA_PATH)/wythoff/exp40
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp40/grid.csv --num_gpu=4 \
		--learning_rate='(0.002, 0.0000001, 10)' \
		--epsilon='(0.5, 0.05, 10)' 
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp40/exp40.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=2000 --batch_size=100 --memory_capacity=10000 --learning_rate={learning_rate} --game=Wythoff15x15 --epsilon={epsilon} --anneal=False --gamma=0.5 --debug=False --update_every=10 --save=$(DATA_PATH)/wythoff/exp40/run_{row_code} --save_model=True --debug=False --monitor='('episode', 'loss', 'score')' --device='cuda:{device_code}' --double=True" :::: $(DATA_PATH)/wythoff/exp40/grid.csv
	
# ----------------------------------------------------------------------------
# 12-28-2019
# First MCTS tuning experiment. Test sim length and exploration constant (c)
wythoff_exp41:
	-rm -rf $(DATA_PATH)/wythoff/exp41
	-mkdir $(DATA_PATH)/wythoff/exp41
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp41/grid.csv \
		--c='(0.041, 2.41, 20)' \
		--num_simulations='(100, 10000, 10)' 
	parallel -j 40 -v \
		--joblog '$(DATA_PATH)/wythoff/exp41/exp41.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_mcts --num_episodes=100 --c={c} --num_simulations={num_simulations} --game=Wythoff15x15 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp41/run_{row_code} --debug=False --monitor='('episode', 'score')'" :::: $(DATA_PATH)/wythoff/exp41/grid.csv

# 12-30-2019
# da4ba0d257fb3894fc4d2c561042aaa43c1de090
#
# use_history=True. history now only saves if the score gets better.
wythoff_exp42:
	-rm -rf $(DATA_PATH)/wythoff/exp42
	-mkdir $(DATA_PATH)/wythoff/exp42
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp42/grid.csv \
		--c='(0.041, 2.41, 20)' \
		--num_simulations='(100, 10000, 10)' 
	parallel -j 40 -v \
		--joblog '$(DATA_PATH)/wythoff/exp42/exp42.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_mcts --num_episodes=100 --c={c} --num_simulations={num_simulations} --game=Wythoff15x15 --debug=False --use_history=True --update_every=1 --save=$(DATA_PATH)/wythoff/exp42/run_{row_code} --debug=False --monitor='('episode', 'score')'" :::: $(DATA_PATH)/wythoff/exp42/grid.csv


# ----------------------------------------------------------------------------
# 1/2/2020
# HP robustness testing.
# ff391749fbe28ed89f22087e4eae7051b1736de3
#
# In exp40 I did grid search for dqn3 performance. Here we test the robustness
# of the top three models.

# ('score', 'learning_rate', 'epsilon')
# (0.9393386211726319, 0.000889, 0.1)
wythoff_exp43:
	-rm -rf $(DATA_PATH)/wythoff/exp43
	-mkdir $(DATA_PATH)/wythoff/exp43
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp43/exp43.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=2000 --batch_size=100 --memory_capacity=10000 --learning_rate=0.000889 --game=Wythoff15x15 --epsilon=0.1 --anneal=False --gamma=0.5 --debug=False --update_every=10 --save=$(DATA_PATH)/wythoff/exp43/run_{1} --save_model=True --debug=False --monitor='('episode', 'loss', 'score')' --device='cuda:0' --double=True" ::: {1..20}
	
# ('score', 'learning_rate', 'epsilon')
# (0.878515854265969, 0.000222, 0.3)
wythoff_exp44:
	-rm -rf $(DATA_PATH)/wythoff/exp44
	-mkdir $(DATA_PATH)/wythoff/exp44
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp44/exp44.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=2000 --batch_size=100 --memory_capacity=10000 --learning_rate=0.000222 --game=Wythoff15x15 --epsilon=0.3 --anneal=False --gamma=0.5 --debug=False --update_every=10 --save=$(DATA_PATH)/wythoff/exp44/run_{1} --save_model=True --debug=False --monitor='('episode', 'loss', 'score')' --device='cuda:1' --double=True" ::: {1..20}
	
# ('score', 'learning_rate', 'epsilon')
# (0.8610649179784786, 0.001111, 0.05)
#
# RESULT: exp45 had the most robust replication. Lower exploration noise?
#         try a run w/ anneal=True?
wythoff_exp45:
	-rm -rf $(DATA_PATH)/wythoff/exp45
	-mkdir $(DATA_PATH)/wythoff/exp45
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp45/exp45.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=2000 --batch_size=100 --memory_capacity=10000 --learning_rate=0.001111 --game=Wythoff15x15 --epsilon=0.05 --anneal=False --gamma=0.5 --debug=False --update_every=10 --save=$(DATA_PATH)/wythoff/exp45/run_{1} --save_model=True --debug=False --monitor='('episode', 'loss', 'score')' --device='cuda:2' --double=True" ::: {1..20}

# 1-2-2020	
# exp44 but with anneal on.
# RESULT: Some increase in consistency and average score distribution.
#         Still quite variable. Should I run a new sweep with anneal=True?
wythoff_exp46:
	-rm -rf $(DATA_PATH)/wythoff/exp46
	-mkdir $(DATA_PATH)/wythoff/exp46
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp46/exp46.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=2000 --batch_size=100 --memory_capacity=10000 --learning_rate=0.000222 --game=Wythoff15x15 --epsilon=0.3 --anneal=True --gamma=0.5 --debug=False --update_every=10 --save=$(DATA_PATH)/wythoff/exp46/run_{1} --save_model=True --debug=False --monitor='('episode', 'loss', 'score')' --device='cuda:1' --double=True" ::: {1..20}

# exp45 but with anneal=True
# RESULT: better average performance than 45. Stabilized performance/curves.
#         With anneal on the final/mean/max results are clearly determined
#         by the initial performance.
wythoff_exp47:
	-rm -rf $(DATA_PATH)/wythoff/exp47
	-mkdir $(DATA_PATH)/wythoff/exp47
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp47/exp47.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=2000 --batch_size=100 --memory_capacity=10000 --learning_rate=0.001111 --game=Wythoff15x15 --epsilon=0.05 --anneal=True --gamma=0.5 --debug=False --update_every=10 --save=$(DATA_PATH)/wythoff/exp47/run_{1} --save_model=True --debug=False --monitor='('episode', 'loss', 'score')' --device='cuda:2' --double=True" ::: {1..20}

# Run exp47 but with a higher intial epsilon=0.5. I'm curious. No hunch. 
# RESULT: Worse than exp47
wythoff_exp48:
	-rm -rf $(DATA_PATH)/wythoff/exp48
	-mkdir $(DATA_PATH)/wythoff/exp48
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp48/exp48.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=2000 --batch_size=100 --memory_capacity=10000 --learning_rate=0.001111 --game=Wythoff15x15 --epsilon=0.5 --anneal=True --gamma=0.5 --debug=False --update_every=10 --save=$(DATA_PATH)/wythoff/exp48/run_{1} --save_model=True --debug=False --monitor='('episode', 'loss', 'score')' --device='cuda:0' --double=True" ::: {1..20}

# ----------------------------------------------------------------------------
# 1-15-20
# Test of alphazero. Run it long to see what happens.
#
# RESULT: Peak optimal play was about 0.4 and that happened after about 
#         500 episodes.
#         After that there was decline. The final score was 0.3352664008698804 
#         with a loss of 0.08834809809923172.
# 
# 		  Going to need to quite a few runs before I get sense of the 
#         possibilites. Long training is not needed. Which is good.
wythoff_exp49:
	-rm -rf $(DATA_PATH)/wythoff/exp49
	-mkdir $(DATA_PATH)/wythoff/exp49
	run_azad.py wythoff_alphazero --num_episodes=1e4 --batch_size=100 --c=0.5 --debug=True --save=$(DATA_PATH)/wythoff/exp49  --game='Wythoff15x15' --max_size=15 --device='cuda:0' > $(DATA_PATH)/wythoff/exp49/debug.log

# First tune sweep for alphazero. Short trian time. Advantage hunting.
# 
# RESULT: a 'c' of ~1.5 gave consistently better score though noting 
#         cracked 0.4 as the median. Poor. learning_rate doesn't show much.
wythoff_exp50:
	-rm -rf $(DATA_PATH)/wythoff/exp50
	-mkdir $(DATA_PATH)/wythoff/exp50
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp50/grid.csv --num_gpu=4 \
		--c='(0.041, 2.41, 20)' \
		--learning_rate='(0.01, 0.00001, 20)' 
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp50/exp50.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_alphazero --num_episodes=1000 --c={c} --learning_rate={learning_rate} --game=Wythoff15x15 --max_size=15 --debug=True --save=$(DATA_PATH)/wythoff/exp50/run_{row_code} --monitor='('episode', 'loss', 'score')' --device='cuda:{device_code}'" :::: $(DATA_PATH)/wythoff/exp50/grid.csv

# ----------------------------------------------------------------------------
# 1/16/2020
# Run DQN3 tune with a conv net. Used a MLP previously.
#
# RESULTS: Best final score was ~0.28. Conv looks terrible.
#          This does make sense. Optimal play needs holistic view of the
#          board in Wythoffs. 
#          Presently the ANN in alphazero uses a ConvNet. Is this performance
#          limiting there? Try MLP.
wythoff_exp51:
	-rm -rf $(DATA_PATH)/wythoff/exp51
	-mkdir $(DATA_PATH)/wythoff/exp51
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp51/grid.csv --num_gpu=4 \
		--learning_rate='(0.002, 0.0000001, 10)' \
		--epsilon='(0.5, 0.05, 10)' 
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp51/exp51.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=2000 --batch_size=100 --memory_capacity=10000 --learning_rate={learning_rate} --game=Wythoff15x15 --network=DQN --epsilon={epsilon} --anneal=False --gamma=0.5 --debug=False --update_every=10 --save=$(DATA_PATH)/wythoff/exp51/run_{row_code} --save_model=True --debug=False --monitor='('episode', 'loss', 'score')' --device='cuda:{device_code}' --double=True" :::: $(DATA_PATH)/wythoff/exp51/grid.csv

# ----------------------------------------------------------------------------
# 1/20/2020
# Try a version of AZ using a simple MLP network instead of the ResNet.
# For motivation see exp51
# 
# RESULT: no improvment compared to ResNet. Final error was still 0.3-5.
#         That seems too high. This error is driven by estimating p (seen
#         by manual testing outside the Makefile). Are my calcs here right?
wythoff_exp52:
	-rm -rf $(DATA_PATH)/wythoff/exp52
	-mkdir $(DATA_PATH)/wythoff/exp52
	run_azad.py wythoff_alphazero --num_episodes=1e4 --batch_size=100 --c=0.5 --debug=True --save=$(DATA_PATH)/wythoff/exp52  --game='Wythoff15x15' --max_size=15 --device='cuda:0' --network_type='MLP' > $(DATA_PATH)/wythoff/exp52/debug.log


# ----------------------------------------------------------------------------
# 1/20/2019
#
# DQN and AZ - some thoughts.
#
# Running out time to finish reviews. I want to add both DQN and MCTS/AZ to 
# the paper. 
#
# - DQN is noisy* but does seem to learn well on occasion. * Noisy: dependent 
#   on a good initial random seed.
# - MCTS (rollout, no ANN) is stuck at 0.4-5 optimality. This is toooooo looow.
#   It should learn to be consistently perfect on Wythoff's RIGHT? 
# - Is there something off w/ my MCTS implementation? Try another persons? 
#   Or try some simple games and prove it out on that? 
# - AZ is stuck at 0.3-4. Less than MCTS. This could be an artifact of
#	the problem w/ MCTS. Or?

# ----------------------------------------------------------------------------
# b8432cb9abcf21a3139b018a4410dc2bca4ca976
# 1/21/2019
# Repeat of exp45 but using modified opt score estimation. Is it consistent?
#
# RESULT: All prev scores for dqn3 were wrong. The opt score code was in error.
#         The analysis of exp51 is especially wrong. The ConvNet did no worse
#         than the MLP in ths experiment. 
wythoff_exp53:
	-rm -rf $(DATA_PATH)/wythoff/exp53
	-mkdir $(DATA_PATH)/wythoff/exp53
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp53/exp53.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=2000 --batch_size=100 --memory_capacity=10000 --learning_rate=0.001111 --game=Wythoff15x15 --epsilon=0.05 --anneal=False --gamma=0.5 --debug=False --update_every=10 --save=$(DATA_PATH)/wythoff/exp53/run_{1} --save_model=True --debug=False --monitor='('episode', 'loss', 'score')' --device='cuda:2' --double=True" ::: {1..20}

# Run a 'good' tuned MCTS game in replicate. Confirm this again with the 
# current code
# c = 1.26, num_simulations=2000
#
# RESULT: consistent w/ previous runs. Expected final score was 0.48.
wythoff_exp54:
	-rm -rf $(DATA_PATH)/wythoff/exp54
	-mkdir $(DATA_PATH)/wythoff/exp54
	parallel -j 20 -v \
		--joblog '$(DATA_PATH)/wythoff/exp54/exp54.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_mcts --num_episodes=100 --c=1.26 --num_simulations=2000 --game=Wythoff15x15 --debug=False --use_history=True --update_every=1 --save=$(DATA_PATH)/wythoff/exp54/run_{} --monitor='('episode', 'score')'" ::: {1..20}

# --------------------------------------------------------------------------
# 1-23-2019
# 18e9f451b14450c33db5df79a05c6e32d94221ce
#
# Final DQN runs for NBDT rev 1.
# 
# We say: DQN does not learn optimal play but it does learn stable Q values that
# can defeat a random player consistently.
#
# NOTE: exp55-58 are setup to run in parallel on seperate gpus (4)
#
# ep=0.1, anneal=False, lr=0.0008, DQN
wythoff_exp55:
	-rm -rf $(DATA_PATH)/wythoff/exp55
	-mkdir $(DATA_PATH)/wythoff/exp55
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp55/exp55.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0008 --game=Wythoff15x15 --epsilon=0.1 --anneal=False --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp55/run_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:0' --double=True --network=DQN --return_none=True" ::: {1..22}

# ep=0.5, anneal=True, lr=0.0008, DQN
wythoff_exp56:
	-rm -rf $(DATA_PATH)/wythoff/exp56
	-mkdir $(DATA_PATH)/wythoff/exp56
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp56/exp56.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0008 --game=Wythoff15x15 --epsilon=0.5 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp56/run_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:1' --double=True --network=DQN --return_none=True" ::: {1..22}

# ep=0.1, anneal=False, lr=0.0008, DQN_mlp
wythoff_exp57:
	-rm -rf $(DATA_PATH)/wythoff/exp57
	-mkdir $(DATA_PATH)/wythoff/exp57
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp57/exp57.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0008 --game=Wythoff15x15 --epsilon=0.1 --anneal=False --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp57/run_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:2' --double=True --network=DQN_mlp --return_none=True" ::: {1..22}

# ep=0.5, anneal=True, lr=0.0008, DQN_mlp
wythoff_exp58:
	-rm -rf $(DATA_PATH)/wythoff/exp58
	-mkdir $(DATA_PATH)/wythoff/exp58
	parallel -j 2 -v \
		--joblog '$(DATA_PATH)/wythoff/exp58/exp58.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0008 --game=Wythoff15x15 --epsilon=0.5 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp58/run_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:3' --double=True --network=DQN_mlp --return_none=True" ::: {1..22}

# ---------------------------------------------------------------------------
# 1-23-2019
# 6000a9b9ebdf3d67c37146c4784aa00c1a0eaf23
#
# Eval the final models from exp55-58 versus self and random play
wythoff_eval55:
	parallel -j 40 -v \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_dqn3 $(DATA_PATH)/wythoff/exp55/run_{}.pytorch --game=Wythoff15x15 --num_episodes=1e2 --opponent=self --debug=False --save=$(DATA_PATH)/wythoff/exp55/eval_self_{} --monitor='('episode', 'total_reward')' --network=DQN --return_none=True" ::: {2..22}
	parallel -j 40 -v \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_dqn3 $(DATA_PATH)/wythoff/exp55/run_{}.pytorch --game=Wythoff15x15 --num_episodes=1e2 --opponent=random --debug=False --save=$(DATA_PATH)/wythoff/exp55/eval_random_{} --monitor='('episode', 'total_reward')' --network=DQN --return_none=True" ::: {2..22}

wythoff_eval56:
	parallel -j 40 -v \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_dqn3 $(DATA_PATH)/wythoff/exp56/run_{}.pytorch --game=Wythoff15x15 --num_episodes=1e2 --opponent=self --debug=False --save=$(DATA_PATH)/wythoff/exp56/eval_self_{} --monitor='('episode', 'total_reward')' --network=DQN --return_none=True" ::: {2..22}
	parallel -j 40 -v \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_dqn3 $(DATA_PATH)/wythoff/exp56/run_{}.pytorch --game=Wythoff15x15 --num_episodes=1e2 --opponent=random --debug=False --save=$(DATA_PATH)/wythoff/exp56/eval_random_{} --monitor='('episode', 'total_reward')' --network=DQN --return_none=True" ::: {2..22}

wythoff_eval57:
	parallel -j 40 -v \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_dqn3 $(DATA_PATH)/wythoff/exp57/run_{}.pytorch --game=Wythoff15x15 --num_episodes=1e2 --opponent=self --debug=False --save=$(DATA_PATH)/wythoff/exp57/eval_self_{} --monitor='('episode', 'total_reward')' --network=DQN_mlp --return_none=True" ::: {2..22}
	parallel -j 40 -v \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_dqn3 $(DATA_PATH)/wythoff/exp57/run_{}.pytorch --game=Wythoff15x15 --num_episodes=1e2 --opponent=random --debug=False --save=$(DATA_PATH)/wythoff/exp57/eval_random_{} --monitor='('episode', 'total_reward')' --network=DQN_mlp --return_none=True" ::: {2..22}

wythoff_eval58:
	parallel -j 40 -v \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_dqn3 $(DATA_PATH)/wythoff/exp58/run_{}.pytorch --game=Wythoff15x15 --num_episodes=1e2 --opponent=self --debug=False --save=$(DATA_PATH)/wythoff/exp58/eval_self_{} --monitor='('episode', 'total_reward')' --network=DQN_mlp --return_none=True" ::: {2..22}
	parallel -j 40 -v \
		--nice 19 --delay 2 \
		"run_azad.py evaluate_dqn3 $(DATA_PATH)/wythoff/exp58/run_{}.pytorch --game=Wythoff15x15 --num_episodes=1e2 --opponent=random --debug=False --save=$(DATA_PATH)/wythoff/exp58/eval_random_{} --monitor='('episode', 'total_reward')' --network=DQN_mlp --return_none=True" ::: {2..22}


# --------------------------------------------------------------------------
# I realized that SS models were run out to 750000 eps. Repeat exp56 and 8
# to that many. (It is not going to matter).
wythoff_exp59:
	-rm -rf $(DATA_PATH)/wythoff/exp59
	-mkdir $(DATA_PATH)/wythoff/exp59
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp59/exp59.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=75000 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0008 --game=Wythoff15x15 --epsilon=0.5 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp59/run_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:1' --double=True --network=DQN --return_none=True" ::: {1..22}

wythoff_exp60:
	-rm -rf $(DATA_PATH)/wythoff/exp60
	-mkdir $(DATA_PATH)/wythoff/exp60
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp60/exp60.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=75000 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0008 --game=Wythoff15x15 --epsilon=0.5 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp60/run_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:3' --double=True --network=DQN_mlp --return_none=True" ::: {1..22}


# --------------------------------------------------------------------------
# Reviewer response 2
# 0124f2821e78a494994966555da624695dd1e6a8
# Many DQN experiments have been done by both Alp and I. Let's refine and 
# collect them into a new figure, for R2.
#
# I first made a bunch of new networks. I've tried these before, or similiar
# but let's put them all toghether and give 'em names.
#
# DQN_hot1, ...
# DQN_xy1, ...
# DQN_conv1, ...
# 
# THere are total of 12 candidates. Let's tune their lrs using a grid search.
#
# Generate the grid to HP search. 
# We'll share it for all these networks.
#
# RESULTS: Analyzed in `wythoff_exp61.ipynb`. The best HP were:
#
# DQN_hot1:
#    row_code  device_code  epsilon  learning_rate
# 1         1            1      0.5         0.0102
# ---
# DQN_hot2:
#      row_code  device_code  epsilon  learning_rate
# 301       301            1      0.2         0.0102
# ---
# DQN_hot3:
#      row_code  device_code  epsilon  learning_rate
# 202       202            2      0.3         0.0203
# ---
# DQN_hot4:
#      row_code  device_code  epsilon  learning_rate
# 201       201            1      0.3         0.0102
# ---
# DQN_hot5:
#      row_code  device_code  epsilon  learning_rate
# 401       401            1      0.1         0.0102
# ---
# DQN_conv1:
#      row_code  device_code  epsilon  learning_rate
# 303       303            3      0.2         0.0304
# ---
# DQN_conv2:
#    row_code  device_code  epsilon  learning_rate
# 1         1            1      0.5         0.0102
# ---
# DQN_conv3:
#      row_code  device_code  epsilon  learning_rate
# 100       100            0      0.4         0.0001
wythoff_exp61:
	-rm -rf $(DATA_PATH)/wythoff/exp61
	-mkdir $(DATA_PATH)/wythoff/exp61
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp61/grid.csv --num_gpu=4 \
		--learning_rate='(0.0001, 1.0, 100)' \
		--epsilon='(0.5, 0.1, 5)'
	parallel -j 32 -v \
		--joblog '$(DATA_PATH)/wythoff/exp61/exp61.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --save=$(DATA_PATH)/wythoff/exp61/run_{1}_{row_code} --num_episodes=5000 --learning_rate={learning_rate} --epsilon={epsilon} --anneal=True --gamma=0.5 --game=Wythoff15x15 --debug=False --network={1} --update_every=10 --monitor='('episode', 'loss', 'score')' --device='cuda:{device_code}'" ::: DQN_hot1 DQN_hot1 DQN_hot2 DQN_hot3 DQN_hot4 DQN_hot5 DQN_conv1 DQN_conv2 DQN_conv3 :::: \
		$(DATA_PATH)/wythoff/exp61/grid.csv

# xy tuning. (conv makes no sense for this)
wythoff_exp62:
	-rm -rf $(DATA_PATH)/wythoff/exp62
	-mkdir $(DATA_PATH)/wythoff/exp62
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp62/grid.csv --num_gpu=4 \
		--learning_rate='(0.0001, 1.0, 100)' \
		--epsilon='(0.5, 0.1, 5)'
	parallel -j 32 -v \
		--joblog '$(DATA_PATH)/wythoff/exp62/exp62.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn2 --save=$(DATA_PATH)/wythoff/exp62/run_{1}_{row_code} --num_episodes=5000 --learning_rate={learning_rate} --epsilon={epsilon} --anneal=True --gamma=0.5 --game=Wythoff15x15 --debug=False --network={1} --update_every=10 --monitor='('episode', 'loss', 'score')' --device='cuda:{device_code}'" ::: DQN_xy1 DQN_xy1 DQN_xy2 DQN_xy3 DQN_xy4 DQN_xy5 :::: \
		$(DATA_PATH)/wythoff/exp62/grid.csv

# -------------------------------------------------------------------------
# 5-7-2020
# d2e1790eb26c356ea03737722f0293b670406355
#
# Run top 1 models from exp61. 
wythoff_exp63a:
	-rm -rf $(DATA_PATH)/wythoff/exp63
	-mkdir $(DATA_PATH)/wythoff/exp63
	# DQN_hot1:
	#    row_code  device_code  epsilon  learning_rate
	# 1         1            1      0.5         0.0102
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp63/exp63.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0102 --game=Wythoff15x15 --epsilon=0.5 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp63/run_DQN_hot1_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:0' --double=True --network=DQN_hot1 --return_none=True" ::: {1..22}

wythoff_exp63b:
	# DQN_hot2:
	#      row_code  device_code  epsilon  learning_rate
	# 301       301            1      0.2         0.0102
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp63/exp63.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
	"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0102 --game=Wythoff15x15 --epsilon=0.2 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp63/run_DQN_hot2_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:1' --double=True --network=DQN_hot2 --return_none=True" ::: {1..22}

wythoff_exp63c:
	# DQN_hot3:
	#      row_code  device_code  epsilon  learning_rate
	# 202       202            2      0.3         0.0203
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp63/exp63.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0203 --game=Wythoff15x15 --epsilon=0.3 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp63/run_DQN_hot3_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:2' --double=True --network=DQN_hot3 --return_none=True" ::: {1..22}

wythoff_exp63d:
	# DQN_hot4:
	#      row_code  device_code  epsilon  learning_rate
	# 201       201            1      0.3         0.0102
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp63/exp63.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0102 --game=Wythoff15x15 --epsilon=0.3 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp63/run_DQN_hot4_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:3' --double=True --network=DQN_hot4 --return_none=True" ::: {1..22}

wythoff_exp63e:
	# DQN_hot5:
	#      row_code  device_code  epsilon  learning_rate
	# 401       401            1      0.1         0.0102
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp63/exp63.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0102 --game=Wythoff15x15 --epsilon=0.1 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp63/run_DQN_hot5_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:0' --double=True --network=DQN_hot5 --return_none=True" ::: {1..22}

wythoff_exp63f:
	# DQN_conv1:
	#      row_code  device_code  epsilon  learning_rate
	# 303       303            3      0.2         0.0304
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp63/exp63.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0304 --game=Wythoff15x15 --epsilon=0.2 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp63/run_DQN_conv1_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:1' --double=True --network=DQN_conv1 --return_none=True" ::: {1..22}

wythoff_exp63f:
	# DQN_conv2:
	#    row_code  device_code  epsilon  learning_rate
	# 1         1            1      0.5         0.0102
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp63/exp63.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0102 --game=Wythoff15x15 --epsilon=0.5 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp63/run_DQN_conv2_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:2' --double=True --network=DQN_conv2 --return_none=True" ::: {1..22}

wythoff_exp63g:
	# DQN_conv3:
	#      row_code  device_code  epsilon  learning_rate
	# 100       100            0      0.4         0.0001
	parallel -j 4 -v \
		--joblog '$(DATA_PATH)/wythoff/exp63/exp63.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn3 --num_episodes=1e3 --batch_size=50 --memory_capacity=10000 --learning_rate=0.0001 --game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --debug=False --update_every=1 --save=$(DATA_PATH)/wythoff/exp63/run_DQN_conv3_{} --debug=False --monitor='('episode', 'loss', 'score', 'Q', 'prediction_error', 'advantage', 'epsilon_e')' --device='cuda:3' --double=True --network=DQN_conv3 --return_none=True" ::: {1..22}
	
# ------------------------------------------------------------------------
# 5-11-2020
# 
# exp62, HP tune using the [s,a -> v] scheme shows actual progress! Last time
# I tried this, that was not true because learning was not at all stable.
# I have no idea what changed this time, but let's take it. 
# 
# BUT
# 
# All the [s,a -> v] are learning quickly, but all of the peak at about 30%
# optimal play. If it can get to 30 it should go all the way? 
# Debug time...

# -
# ffe1c2be88b674d1050d5658d0e196743ef57409
# First, the way optimal is calculated uses slightly different code. In exp64
# I use the optimal play calc code from SS. A first control. 
# Also
# - DQN_xy3 and 4 showed the most promise. 
# - Pick a tune search near exp62's best for these two models. 
# DQN_xy3:
    #  row_code  device_code  epsilon  learning_rate
# 459       459            3      0.1          0.596
# ---
# DQN_xy4:
    #  row_code  device_code  epsilon  learning_rate
# 324       324            0      0.2         0.2425
#
# RESULTS:  It's not scoring (wheew!)
wythoff_exp64:
	-rm -rf $(DATA_PATH)/wythoff/exp64
	-mkdir $(DATA_PATH)/wythoff/exp64
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp64/grid.csv --num_gpu=4 \
		--learning_rate='(0.1, 0.5, 10)' \
		--epsilon='(0.1, 0.4, 5)'
	parallel -j 32 -v \
		--joblog '$(DATA_PATH)/wythoff/exp64/exp64.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn2 --save=$(DATA_PATH)/wythoff/exp64/run_{1}_{row_code} --num_episodes=5000 --learning_rate={learning_rate} --epsilon={epsilon} --anneal=True --gamma=0.5 --game=Wythoff15x15 --debug=False --network={1} --update_every=10 --monitor='('episode', 'loss', 'score')' --device='cuda:{device_code}'" ::: DQN_xy3 DQN_xy4 :::: \
		$(DATA_PATH)/wythoff/exp64/grid.csv

# ------------------------------------------------------------------------
# 5-11-2020
# It seems that despite learning optimal play to 0.3 in the best case, the xy
# models also, and at the same time, have an exploding loss/Q value issue.
# This seems to be fixed if gamma < 0.5 instead of 0.5. 
#
# Rerun all the xy tuning, adding a gamma search too.
#
# Other changes: num_episodes=500, update_every=1
#
# RESULTS: Loss explodes, and optimal play improves. Not quite 1:1 but certianly
#          related. How terrible and confusing. I do not know what to do...
wythoff_exp65:
	-rm -rf $(DATA_PATH)/wythoff/exp65
	-mkdir $(DATA_PATH)/wythoff/exp65
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp65/grid.csv --num_gpu=4 \
		--learning_rate='(0.0025, 0.25, 50)' \
		--epsilon='(0.1, 0.5, 5)' \
		--gamma='(0.1, 0.5, 5)' 
	parallel -j 32 -v \
		--joblog '$(DATA_PATH)/wythoff/exp65/exp65.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn2 --save=$(DATA_PATH)/wythoff/exp65/run_{1}_{row_code} --num_episodes=500 --learning_rate={learning_rate} --epsilon={epsilon} --anneal=True --gamma={gamma} --game=Wythoff15x15 --debug=False --network={1} --update_every=1 --monitor='('episode', 'loss', 'score')' --device='cuda:{device_code}'" ::: DQN_xy1 DQN_xy1 DQN_xy2 DQN_xy3 DQN_xy4 DQN_xy5 :::: \
		$(DATA_PATH)/wythoff/exp65/grid.csv

# -------------------------------------------------------------------------
# 5-13-2020
# 955de31d2361629e8b128bfa4e85dc73d7ee818e
#
# Is it grad explosion? I exposed grad_clip. Rerun exp65 w/ clipping.
wythoff_exp66:
	-rm -rf $(DATA_PATH)/wythoff/exp66
	-mkdir $(DATA_PATH)/wythoff/exp66
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp66/grid.csv --num_gpu=4 \
		--learning_rate='(0.0025, 0.25, 50)' \
		--epsilon='(0.1, 0.5, 5)' \
		--gamma='(0.1, 0.5, 5)' 
	parallel -j 32 -v \
		--joblog '$(DATA_PATH)/wythoff/exp66/exp66.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_dqn2 --save=$(DATA_PATH)/wythoff/exp66/run_{1}_{row_code} --num_episodes=250 --learning_rate={learning_rate} --epsilon={epsilon} --anneal=True --clip_grad=False --gamma={gamma} --game=Wythoff15x15 --debug=False --network={1} --update_every=1 --monitor='('episode', 'loss', 'score')' --device='cuda:{device_code}'" ::: DQN_xy1 DQN_xy1 DQN_xy2 DQN_xy3 DQN_xy4 DQN_xy5 :::: \
		$(DATA_PATH)/wythoff/exp66/grid.csv
