import os
import sys
import random
import time

import gym_mdptetris.envs
import numpy as np
from gym_mdptetris.envs import board, piece
from torch.utils.tensorboard import SummaryWriter


class RandomLinearGame():
    def __init__(self, board_height=20, board_width=10,  piece_set='pieces4.dat', seed=12345):
        """
        Class to implement a linear game with random strategy using the structures
        and methods from gym-mdptetris. 
        """
        self.board_height = board_height
        self.board_width = board_width
        path = os.path.dirname(gym_mdptetris.envs.__file__)
        pieces_path = path + '/data/' + piece_set
        self.pieces, self.nb_pieces = piece.load_pieces(pieces_path)
        self.max_piece_height = 0
        for p in self.pieces:
            for o in p.orientations:
                self.max_piece_height = max(self.max_piece_height, o.height)
        random.seed(seed)
        self.new_piece()
        self.board = board.Board(max_piece_height=self.max_piece_height,
                                 width=board_width, height=board_height)

    def new_piece(self):
        """
        Method to select the next piece. 
        """
        self.current_piece = random.choice(range(self.nb_pieces))

    def seed(self, seed_value: int):
        """
        Seed randomness for game. 

        :param seed_value: New seed value for game
        """
        random.seed(seed_value)

    def reset(self):
        """
        Reset the game and return the new board state. 

        return: Current board state
        """
        self.board.reset()
        self.new_piece()
        self.lines_cleared = 0

    def board_step(self):
        """
        Make one random action.

        :return: Returns the lines cleared by the action. 
        """
        a = [random.randint(
            0, self.pieces[self.current_piece].nb_orientations - 1), 0]
        a[1] = random.randint(
            0, self.board_width - self.pieces[self.current_piece].orientations[a[0]].width - 1)
        return self.board.drop_piece(self.pieces[self.current_piece].orientations[a[0]], a[1])

    def play_game(self, render=False):
        """
        Method to play an episode of a random strategy game. 
        """
        cleared = 0
        timesteps = 0
        while self.board.wall_height < self.board_height:
            timesteps += 1
            cleared += self.board_step()
            self.new_piece()
            if render:
                print(self.board)

        return cleared, timesteps


def test_performance(seed: int=12345, nb_games: int=100, log_dir: str='runs', save_dir: str='./'):
    """
    Method to test performance of Dellacherie method.

    :param seed: Seed for the environment 
    :param nb_games: number of episodes to run test for
    :param log_dir: Directory to log TensorBoard results to
    :param save_dir: Directory to save episode reward results to
    """
    runid = "Random" + time.strftime('%Y%m%dT%H%M%SZ')
    writer = SummaryWriter(log_dir, comment=f"Random-{runid}")
    lg = RandomLinearGame()
    episode_rewards = []
    episode_duration = []
    for i in range(nb_games):
        reward, timesteps = lg.play_game()
        print(f"Episode reward: {reward}, episode duration: timesteps")
        episode_rewards.append(reward)
        episode_duration.append(timesteps)
        lg.reset()
        writer.add_scalar(f"Random-{runid}/Episode reward", reward, i)
        writer.add_scalar(f"Random-{runid}/Episode duration", timesteps, i)

    np.array(episode_rewards).tofile(f"{save_dir}/Random-rewards-{runid}.csv", sep=',')
    np.array(episode_duration).tofile(f"{save_dir}/Random-timesteps-{runid}.csv", sep=',')
    print(f"Average rewards: {np.mean(np.array(episode_rewards))}")
    print(f"Average duration: {np.mean(np.array(episode_duration))}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            nb_games = 10000
            if len(sys.argv) > 2:
                nb_games = int(sys.argv[2])
            test_performance(nb_games=nb_games, save_dir="./runs")
            sys.exit(0)
    lg = RandomLinearGame()
    start = time.time()
    lg.play_game()
    end = time.time()
    print(f"That took {end - start}s")
