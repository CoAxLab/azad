# Key points

1. The stumbler-strategist and the usefulness of weak human given heuristics in generalization. Our stumbler teaches a second network, which in time learns to transform the task into supervised learning. In its Deep Form the strategist learns not just the diagonals, but phi itself.

Improves on Raghu et al (2018) by:

1. Stimulus and game generalization (they 'generalize' to sub-optimal opponents).
2. The overall generality/usefulness of the stumbler-strategist, and heuristics.


# Text/figure improvemnents

- Rewrite into stumbler-strategist form
- Rewrite the Wythoff game -- shorten
- Fix readability of Algorithm figs -- shorten
- Better strategy/game figure (current Fig 2.)
- Re-run Fig 5 many times to get error bars; make axis clearer.
- Re-run Fig 6 for error bars.
- Redo Fig 7 design to get rid of serial game effects
- Add Figs for controls (below).
- Add new experiment Figs? (below).


# Experiments

- Control: Stumbler players get seperate tables/models with 'loss' learning for each.
  - Current 'look ahead' desgin is not actually Q learning.
- Control: Try (x,y) representation for stumbler? 
  - Does it generalize on its own now? (It might.)
- Control: Remove symetry w/ sampling. Is this really responsible?
- Control: Do supervised tranining; does the strategist layer help with brittleness here?
  - Is any model learning to 'go cold'? That is the optimal play.
- Control: How much does an _optimal move strategist_ help; a positive control. 
  - Add to all (relevant) figures?
- Control: Parameter sensitivity testing for Hot/Cold projection
- Control: Learning rate sensitivity

- New: Try a 3d board? 
- New: Drop `L - e^(.)`, stumbler influence should be incremental and performance based
  - Use the `learning_rate` to increment when it is better?
- New: Compare deep to shallow strategist.


# Code
## General API
- Update stumbler DNN code to match tabular code
  - Confirm learning
  - Default is now the DeepStumbler but,
    - keep the tabular around as positive control
- Update strategist to work w/ above
  - Confirm learning
  - Performance/incremental effect implementation

- Drop look ahead learning, moving to:
- `Agent(...)` refactor? 
  - `player = Agent(env, done, ...)`; `opponent = Agent(env, done, ...)`
  - ... w/ other player loss learning at the end
  - `player = Strategist(env, stumbler_agent)`
  - `TwoPlayerGame(env, player, opponent)`
  - Random/greedy opponents (that don't learn themselves)
  - Implement `Agent(.).save(name)` and `Agent(.).load(name)`.
  
  - Implement human / CL player option
    - Show on CL as piles not boards? (Piles allow for N-dim games).
    - Allow for play against a trained Agent(.)

## Current results
- Perf. based iteration for strategists 'assistance'?
- Nym implementation
- Euclid implementation
- Do: Hot/Cold tuning/parameterization.
- In Makefile, create exp designs for Fig updates/regeneration of Alps

## New results/controls
- Implement N-dim Wythoffs (at least 3d).
- Implement Deep (x,y) stumbler
- Implement option for no 'symmetric' sampling of Cold places
- Implement supervised/optimal move strategist layer
- Implement move analysis code; Do the models learn to leave their opponents in Cold places? (This is optimal play.) 