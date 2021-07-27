import argparse
from mdptetris_experiments.agents.action_networks import NN1DAction, NNHeuristicAction

from mdptetris_experiments.agents.FFNN import NNHeuristic, NN1D
from gym_mdptetris.envs.tetris import TetrisFlat
from mdptetris_experiments.agents.linear_agent import LinearGame, LinearGameStandard, MultiLinearGame
from train import PPO

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0',
                        help="The GPU to train the agent on")
    parser.add_argument("--board_height", type=int, default=20)
    parser.add_argument("--board_width", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--batch_timesteps", type=int, default=4500)
    parser.add_argument("--max_episode_timesteps", type=int, default=2000)
    parser.add_argument("--nb_games", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--saving_interval", type=int, default=500)
    parser.add_argument("--state_rep", type=str, default="heuristic")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--load_file", type=str, default=None,
                        help="Path to partially trained model")
    parser.add_argument("--save_dir", type=str,
                        default=f"runs/run-info")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--comment", type=str, default="test",
                        help="Run comment for TensorBoard writer.")

    args = parser.parse_args()
    return args

state_rep = {
    "heuristic": [NNHeuristic, MultiLinearGame],
    "1D": [NN1D, LinearGameStandard]
}

def main():
    args = get_args()

    env = TetrisFlat(board_height=args.board_height,
                     board_width=args.board_width, seed=args.seed)

    args = vars(args)
    model = PPO(args, NN1DAction, env)
    print("Model and env initialised")
    model.train(200_000_000)

if __name__=='__main__':
    main()


