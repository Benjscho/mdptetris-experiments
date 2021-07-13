# Commands to launch experiments

## Heuristic DQN, longer learning period 
`python mdptetris_experiments/agents/DQN/train.py --epochs=10000 --epsilon_decay_period=6000`

## Flattened array DQN
`python mdptetris_experiments/agents/DQN/train.py --state_rep="1D"`