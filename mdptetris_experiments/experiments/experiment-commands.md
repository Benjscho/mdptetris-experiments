# Commands to launch experiments

## Heuristic DQN, longer learning period 
`python mdptetris_experiments/agents/DQN/train.py --epochs=10000 --epsilon_decay_period=6000`

`python mdptetris_experiments/agents/DQN/train.py --epochs=10000 --epsilon_decay_period=6000 --board_height=10`

## Flattened array DQN
`python mdptetris_experiments/agents/DQN/train.py --state_rep="1D"`

`python mdptetris_experiments/agents/DQN/train.py --state_rep="1D" --epochs=10000 --epsilon_decay_period=6000`

`python mdptetris_experiments/agents/DQN/train.py --state_rep="1D" --epochs=10000 --epsilon_decay_period=6000 --board_height=10`