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
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp14/run_{1} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}


# --- Heuristic controls
# SS w/ only cold
wythoff_exp15:
	-rm -rf $(DATA_PATH)/wythoff/exp15
	-mkdir $(DATA_PATH)/wythoff/exp15
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp15/exp15.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp15/run_{1} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=None --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# SS w/ only hot
wythoff_exp16:
	-rm -rf $(DATA_PATH)/wythoff/exp16
	-mkdir $(DATA_PATH)/wythoff/exp16
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp16/exp16.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
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
		--nice 19 --delay 2 --header : --colsep ',' \
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
		--nice 19 --delay 2 --header : --colsep ',' \
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
wythoff_exp24:
	-rm -rf $(DATA_PATH)/wythoff/exp24
	-mkdir $(DATA_PATH)/wythoff/exp24
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp24/exp24.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py evaluate_wythoff --save=$(DATA_PATH)/wythoff/exp24/run_{1}_{2} --load_model=$(DATA_PATH)/wythoff/exp14/run_{1}.pytorch --num_episodes=1000 --strategist_game={2} --stumbler_game=Wythoff15x15 --return_none=True" ::: \
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
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp25/run_{1} --monitor='('episode', 'influence', 'eval_score_a', 'eval_score_b')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Nim15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Nim50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# Euclid
wythoff_exp26:
	-rm -rf $(DATA_PATH)/wythoff/exp26
	-mkdir $(DATA_PATH)/wythoff/exp26
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp26/exp26.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp26/run_{1} --monitor='('episode', 'influence', 'eval_score_a', 'eval_score_b')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Euclid15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Euclid50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# 2. Pre-train Wythoff (exp14): Train Nim, Euclid
# Nim
wythoff_exp27:
	-rm -rf $(DATA_PATH)/wythoff/exp27
	-mkdir $(DATA_PATH)/wythoff/exp27
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp27/exp27.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp27/run_{1} --load_model=$(DATA_PATH)/wythoff/exp14/run_{1}.pytorch --monitor='('episode', 'influence', 'eval_score_a', 'eval_score_b')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Nim15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Nim50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
		{1..20}

# Euclid
wythoff_exp28:
	-rm -rf $(DATA_PATH)/wythoff/exp28
	-mkdir $(DATA_PATH)/wythoff/exp28
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp28/exp28.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp28/run_{1} --load_model=$(DATA_PATH)/wythoff/exp14/run_{1}.pytorch --monitor='('episode', 'influence', 'eval_score_a', 'eval_score_b')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Nim15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Nim50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" ::: \
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
		--hot_threshold='(0.0, 0.4, 10)' \
		--cold_threshold='(0.0, -0.4, 10)'
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp33/exp33.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp33/run_{row_code} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon=0.4 --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold={cold_threshold} --hot_threshold={hot_threshold} --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" :::: \
		$(DATA_PATH)/wythoff/exp33/grid.csv

# epsilon
wythoff_exp34:
	-rm -rf $(DATA_PATH)/wythoff/exp34
	-mkdir $(DATA_PATH)/wythoff/exp34
	run_azad.py create_grid $(DATA_PATH)/wythoff/exp34/grid.csv \
		--epsilon='(0.01, 0.8, 20)' 
	parallel -j 8 -v \
		--joblog '$(DATA_PATH)/wythoff/exp34/exp34.parallel.log' \
		--nice 19 --delay 2 --header : --colsep ',' \
		"run_azad.py wythoff_stumbler_strategist --save=$(DATA_PATH)/wythoff/exp34/run_{row_code} --monitor='('episode', 'influence')' --stumbler_monitor='('episode', 'loss', 'score', 'total_reward')' --strategist_monitor='('episode', 'loss', 'mae')' --num_episodes=150 --update_every=10 --learning_rate_influence=0.2 --num_stumbles=500 --learning_rate_stumbler=0.4 --stumbler_game=Wythoff15x15 --epsilon={epsilon} --anneal=True --gamma=0.5 --num_strategies=500 --learning_rate_strategist=0.025 --strategist_game=Wythoff50x50 --cold_threshold=-0.2 --hot_threshold=0.2 --hot_value=-1 --cold_value=1 --debug=False --save_model=True --return_none=True --debug=False --seed={1}" :::: \
		$(DATA_PATH)/wythoff/exp34/grid.csv