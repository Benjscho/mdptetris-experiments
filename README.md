# MdpTetris experiments

This repository holds the agents and experiments to be run on 
[`gym-mdptetris`](https://github.com/Benjscho/gym-mdptetris).

## Installation

This repository can be installed via GitHub:
```bash
pip install git+https://github.com/Benjscho/mdptetris-experiments
```
Or by cloning to the desired directory, `cd`-ing into the directory and 
running `pip install -e .`.

## Run experiments

### Linear agent

The linear agent can be run with the default weights created by Pierre 
Dellacherie using `python mdptetris_experiments/agents/linear_agent.py render`. 
This agent is very effective at clearing lines, so this can take some
time to run.

This agent is very successful:
<p align="left">
    <img src="assets/dellacherie.gif" width="400">
</p>

To customize the weighting of the agent, a new instance can be created:

```python
import numpy as np
from mdptetris_experiments.agents.linear_agent import LinearGame

agent = LinearGame(weights=np.array([-1, 1, -1, -1, -4, -1]))
cleared = agent.play_game()
print(f"{cleared} rows cleared")
```

