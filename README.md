# Azad

Game-playing ANNs that use a stumbler-strategist architecture. Stumblers learn to map states to actions using model-free RL. Strategists _only_ study stumblers, using the combination of a deep net and a human-derived heuristic to find the structure between the states. Adding in a heuristic means the strategist layer is model-based, and so our overall architecture is related the classic [DYNA-Q](https://medium.com/@ranko.mosic/online-planning-agent-dyna-q-algorithm-and-dyna-maze-example-sutton-and-barto-2016-7ad84a6dc52b) form.

A heuristic--it turns out--leads to robust transfer between games, and lead to natutally interpretable representations.

Philosophically, we are interested in co-operative AI. By sharing strategies and heuristics our ANNs can try and help people, not replace them. 

# Papers

- Peterson, E.J., Muyesser, N.A., Verstynen, T. & Dunovan, K. 2018. Keep it Stupid Simple, ArXiv 1809.03406. Available at: https://arxiv.org/abs/1809.03406.

We tried a {`good`, `bad`} heuristic, where complex `Q(s,a)` values get mapped to either `good` or `bad` classes. We use these classes to transfer knowledge to new games. Using this network we studied a partial combinatorial game, Wythoff's game, and some of its relatives.

# Values

We come at ANN design as scientists and humanists. Meaning we take three strong philosophical stances:

1. ANNs should help people, not replace them. 
2. To win in the long-term ANNs and science must create a close virtuous cycle of improvement. Principled science--psychology, biology, neuroscience--should directly inform ANN design. ANN results should directly inform science.
3. ANNs must help us understand our problems better. This means an ANN should always try to explain itself to a person. 


# Dependencies

- python3
- a standard Anaconda install
- pytorch
- fire (https://github.com/google/python-fire)


## Optional tensorboard visuals

- tensorflow
- tensorboard
- tensorboardX 

Install instructions: https://github.com/lanpa/tensorboard-pytorch


# Install

1. From the command line run `git clone https://github.com/CoAxLab/azad.git`
2. Then from the top-level `azad` directory run, `pip install .` for a normal install or `pip install -e .` if you are going to be editing the code.


# That funny project name

In Ian Banks delightful book, *The Player of Games*, master game player Jernau Morat Gurgeh travels to the planet Azad to play the game Azad. Though it takes a lifetime to master, Azad is as much a statement of philosophy as it is a game of winning and strategy. In the book, Gurgeh comes from an alien culture so his philosophy and play is quite different than his opponents.

