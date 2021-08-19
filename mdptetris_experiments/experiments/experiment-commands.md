# Commands to launch experiments

## MBDQN Model-Based
### Heuristic MBDQN, longer learning period 
`python mdptetris_experiments/agents/MBDQN/train.py --epochs=10000 --epsilon_decay_period=6000`
- Runid: 20210809T092959Z
- Garlick

`python mdptetris_experiments/agents/MBDQN/train.py --epochs=10000 --epsilon_decay_period=6000 --board_height=10`
- Runid: 20210809T094830Z
- Nitt

### Flattened array MBDQN
`python mdptetris_experiments/agents/MBDQN/train.py --state_rep="1D"`

`python mdptetris_experiments/agents/MBDQN/train.py --state_rep="1D" --epochs=10000 --epsilon_decay_period=6000`
- Runid: 20210809T095013Z
- Nitt

`python mdptetris_experiments/agents/MBDQN/train.py --state_rep="1D" --epochs=10000 --epsilon_decay_period=6000 --board_height=10`

## PPO
### PPO Full State Space
`python mdptetris_experiments/agents/PPO2/main.py`

### PPO Small Board
`python mdptetris_experiments/agents/PPO2/main.py --board_height=10`