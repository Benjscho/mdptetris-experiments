import argparse
import torch.multiprocessing as mp

from gym_mdptetris.envs.tetris import TetrisFlat
from mdptetris_experiments.agents.action_networks import NN1DAction

from train import PPO


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0',
                        help="The GPU to train the agent on")
    parser.add_argument("--board_height", type=int, default=20)
    parser.add_argument("--board_width", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--batch_timesteps", type=int, default=10000)
    parser.add_argument("--max_episode_timesteps", type=int, default=2000)
    parser.add_argument("--nb_games", type=int, default=20)
    parser.add_argument("--updates_per_iter", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip", type=float, default=0.2)
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


class MultiEnv:
    def __init__(self, board_height, board_width, seed, nb_envs):
        self.agent_con, self.env_con = zip(*[mp.Pipe() for _ in range(nb_envs)])

        self.envs = []
        for _ in range(nb_envs):
            self.envs.append(TetrisFlat(board_height=board_height,
                     board_width=board_width, seed=seed))
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        for i in range(nb_envs):
            process = mp.Process(target=self.run_env, args=(i,))
            process.start()
            self.env_con[i].close()

    def run_env(self, index):
        self.agent_con[index].close()
        while True:
            request, action = self.env_con[index].recv()
            if request == 'step':
                self.env_con[index].send(self.envs[index].step(action.item()))
            elif request == 'reset':
                self.env_con[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError


def main():
    args = get_args()

    envs = MultiEnv(board_height=args.board_height,
                    board_width=args.board_width, seed=args.seed, nb_envs=args.nb_games)

    args = vars(args)
    model = PPO(args, NN1DAction, envs)
    print("Model and env initialised")
    model.train(200_000_000)


if __name__ == '__main__':
    main()
