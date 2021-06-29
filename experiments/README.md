# Experiment planning
## Environment variables

Variable | nb_variables | Options
---|---|--- 
Height + Width | 3 | (20, 10), (10, 10), (2, 6) 
Pieces | 2 | Standard, Melax
State representation | 4 | (PieceID, Board 2D), (Board inc. Piece 2D), (Board inc. Piece 1D), Heuristic
Reward | 2 | Lines cleared, lines overflowed over X pieces


## Algorithms
- Q-Learning (small board and pieces only)
- DQN
- PPO
- CBMPI
- Linear baseline (Dellacherie features and weights)
- Random agent

## Experiments to run

Algorithm | Height + width | Pieces | State representation | Reward
---|---|---|---|---
Linear baseline | (2, 6) | Melax | Heuristic | lines overflowed over 10,000 pieces
Linear baseline | (10, 10) | Standard | Heuristic | lines cleared
Linear baseline | (20, 10) | Standard | Heuristic | lines cleared
Q-Learning | (2,6) | Melax | (PieceID, Board 2D) | lines overflowed over 10,000 pieces
Q-Learning | (2,6) | Melax | (Board inc. Piece 2D) | lines overflowed over 10,000 pieces
Q-Learning | (2,6) | Melax | (Board inc. Piece 1D) | lines overflowed over 10,000 pieces
Q-Learning | (2,6) | Melax | Heuristic | lines overflowed over 10,000 pieces
DQN | (2,6) | Melax | (Board inc. Piece 2D) | lines overflowed over 10,000 pieces
DQN | (10, 10) | Standard | (PieceID, Board 2D) | lines cleared
DQN | (10, 10) | Standard | (Board inc. Piece 1D) | lines cleared
DQN | (10, 10) | Standard | (Board inc. Piece 2D) | lines cleared
DQN | (10, 10) | Standard | Heuristic | lines cleared
DQN | (20, 10) | Standard | ? - tbd by 10,10 experiments | lines cleared
PPO | (2,6) | Melax | (Board inc. Piece 2D) | lines overflowed over 10,000 pieces
PPO | (10, 10) | Standard | (PieceID, Board 2D) | lines cleared
PPO | (10, 10) | Standard | (Board inc. Piece 1D) | lines cleared
PPO | (10, 10) | Standard | (Board inc. Piece 2D) | lines cleared
PPO | (10, 10) | Standard | Heuristic | lines cleared
PPO | (20, 10) | Standard | ? - tbd by 10,10 experiments | lines cleared


## Data to collect in experiments

Algorithm | Data 
---|---
Linear baseline | Total reward
Random agent | Total reward
Q-Learning | Total reward, episodes trained, loss value
DQN | Total reward, episodes trained, loss value, hyperparameters
PPO | Total reward, episodes trained, loss value, hyperparameters
