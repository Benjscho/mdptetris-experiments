# Experiment planning
## Environment variables

Variable | nb_variables | Options
---|---|--- 
Height + Width | 3 | (20, 10), (10, 10), (2, 6) 
Pieces | 2 | Standard, Melax
State representation | 4 | (PieceID, Board 2D), (Board inc. Piece 2D), (Board inc. Piece 1D), Heuristic
Reward | 2 | Lines cleared, lines overflowed over X pieces


## Potential Algorithms
- Q-Learning (small board and pieces only)
- DQN
- PPO
- CBMPI
- Linear baseline (Dellacherie features and weights)
- Random agent

## Proposed experiments to run

Algorithm | Height + width | Pieces | State representation | Reward | RunID
---|---|---|---|---|---
Linear baseline | (2, 6) | Melax | Heuristic | lines overflowed over 10,000 pieces
DQN | (2,6) | Melax | (Board inc. Piece 2D) | lines overflowed over 10,000 pieces
PPO | (2,6) | Melax | (Board inc. Piece 2D) | lines overflowed over 10,000 pieces
Linear baseline | (10, 10) | Standard | Heuristic | lines cleared
Linear baseline | (20, 10) | Standard | Heuristic | lines cleared
DQN | (10, 10) | Standard | (PieceID, Board 2D) | lines cleared
DQN | (10, 10) | Standard | (Board inc. Piece 1D) | lines cleared | 20210715T112914Z
DQN | (10, 10) | Standard | (Board inc. Piece 2D) | lines cleared
DQN | (10, 10) | Standard | Heuristic | lines cleared | 20210715T113035Z
DQN | (20, 10) | Standard | Heuristic | lines cleared | 
DQN | (20, 10) | Standard | Board 1D | lines cleared | 20210713T101844Z
PPO | (10, 10) | Standard | (PieceID, Board 2D) | lines cleared
PPO | (10, 10) | Standard | (Board inc. Piece 1D) | lines cleared
PPO | (10, 10) | Standard | (Board inc. Piece 2D) | lines cleared
PPO | (10, 10) | Standard | Heuristic | lines cleared
PPO | (20, 10) | Standard | ? - tbd by 10,10 experiments | lines cleared | 20210730T093417Z
Experiments below only if time is available
Q-Learning | (2,6) | Melax | (PieceID, Board 2D) | lines overflowed over 10,000 pieces
Q-Learning | (2,6) | Melax | (Board inc. Piece 2D) | lines overflowed over 10,000 pieces
Q-Learning | (2,6) | Melax | (Board inc. Piece 1D) | lines overflowed over 10,000 pieces
Q-Learning | (2,6) | Melax | Heuristic | lines overflowed over 10,000 pieces

## Model experiments to run

Algorithm | Height + width | Pieces | State representation | Reward | RunID
---|---|---|---|---|---
TD-Lambda | (10, 10) | Standard - 1 per episode| (Board inc. Piece 1D) | lines cleared | 20210715T112914Z
TD-Lambda | (10, 10) | Standard - 1 per episode | Heuristic | lines cleared | 20210715T113035Z
TD-Lambda | (20, 10) | Standard - 1 per episode | (Board inc. Piece 1D) | lines cleared | 20210713T101844Z
TD-Lambda | (20, 10) | Standard - 1 per episode | Heuristic | lines cleared | 20210803T070646Z  
TD-Lambda | (10, 10) | Standard | (Board inc. Piece 1D) | lines cleared | 
TD-Lambda | (10, 10) | Standard | Heuristic | lines cleared | 
TD-Lambda | (20, 10) | Standard | (Board inc. Piece 1D) | lines cleared | 
TD-Lambda | (20, 10) | Standard | Heuristic | lines cleared | 
PPO | (20, 10) | Standard | (Board inc. Piece 1D) | lines cleared | 20210730T093417Z
PPO | (10, 10) | Standard | (Board inc. Piece 1D) | lines cleared | 20210802T205340Z

Just had an accident where the epsilon went to below zero, this highlights I
need to test the trained agents! This DQN is running for ages now. I could be
getting ace performance... Don't just take the results of the training. Verify
with tests of the trained models. 

Found the issue with the "DQN" (TD-Lambda) runs, new pieces were only being selected 
when the episode was reset. I've corrected that now but the performance has been majorly hit. 

## Test demos to run

Algorithm | Height + width | Pieces | State representation | Reward | TrainRunID | TestRunID
---|---|---|---|---|---|---
Linear baseline (Random agent) | (20, 10) | Standard | Heuristic | lines cleared
Linear baseline (Dellacherie weights) | (10, 10) | Standard | Heuristic | lines cleared
Linear baseline (Dellacherie weights) | (20, 10) | Standard | Heuristic | lines cleared | | 20210805T084801Z
TD-Lambda | (10, 10) | Standard - 1 per episode| (Board inc. Piece 1D) | lines cleared | 20210715T112914Z
TD-Lambda | (10, 10) | Standard - 1 per episode | Heuristic | lines cleared | 20210715T113035Z
TD-Lambda | (20, 10) | Standard - 1 per episode | (Board inc. Piece 1D) | lines cleared | 20210713T101844Z
TD-Lambda | (20, 10) | Standard - 1 per episode | Heuristic | lines cleared | 20210803T070646Z | 
PPO | (20, 10) | Standard | (Board inc. Piece 1D) | lines cleared | 20210730T093417Z
PPO | (10, 10) | Standard | (Board inc. Piece 1D) | lines cleared | 20210802T205340Z


## Data to collect in experiments

Algorithm | Data 
---|---
Linear baseline | Total reward
Random agent | Total reward
Q-Learning | Total reward, episodes trained, loss value
DQN | Total reward, time steps trained, loss value, hyperparameters, learning curve of average reward against time steps
PPO | Total reward, time steps trained, loss value, hyperparameters, learning curve of average reward of episodes in a batch

There are a lot of ways of collecting learning curve. PPO can be hard to collect learning curve as it runs multiple envs at once, you can finish half an episode. Taking average return of all episodes within a rollout batch for PPO can be a good metric.

Use time steps instead of episodes trained for the comparison graphs, this allows comparison well for PPO. 
Loss value is useful to collect but may not be included in the paper. It can help you see when the maths is wrong

Hyperparameters stored as a table in appendix. Decisions must be made as to whether hyperparameters are to be fixed or experimented with. Can tune hyperparams with a simple experiment and then fix them for the rest of the experiment, they can be very fiddly. 

The main thing I want to compare is algorithms and state representation. 
Doing anything weird with models can be an experiment. Using neural nets.
Can skip Melax as its not a variable in the experiment much. The pieces aren't constant. 

Would be interesting to explore the reward function, punishing the agent for placing a piece higher up the board, there's no negative reward for placing a piece higher up the board. 

## Main experiment comparison points
- State representation 
- Algorithms 
- If looking for another element to analyse, add in different model representations, CNNs versus MLPs. 