import os
import random
import sys
import time
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter

import gym_mdptetris.envs
import numpy as np
import torch
from gym_mdptetris.envs import board, feature_functions, piece


class LinearGame():
    def __init__(self, weights: np.ndarray = np.array([-1, 1, -1, -1, -4, -1]),
                 board_height: int = 20,
                 board_width: int = 10,
                 piece_set: str = 'pieces4.dat',
                 seed: int = 12345):
        """
        Linear game of Tetris for reinforcement learning.

        By default the feature set and weights are set to those of Pierre Dellacherie.
        This hand tuning is excellent at the game, and runs can last for an extended
        period of time. 

        The state-value of a board is the sum of the product of the state's heuristic
        representation and the weights given to each feature.

        :param weights: A numpy array of weights ascribed to the feature set. 
        :param board_height: The height of the tetris board. 
        :param board_width: The width of the tetris board.
        :param piece_set: Relative path to the piece set in the data directory.
        :param seed: Seed of the random function. 
        """
        self.weights = weights
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
        self.lines_cleared: int = 0
        self.board = board.Board(max_piece_height=self.max_piece_height,
                                 width=board_width, height=board_height)

    def get_state(self) -> torch.FloatTensor:
        """
        Get the heuristic representation of a board state as the set of
        6 Dellacherie features.

        :return: Numpy array of the Dellacherie features of the current
            board state.
        """
        res = []
        for f in feature_functions.get_dellacherie_funcs():
            res.append(f(self.board))
        return torch.FloatTensor(res)

    def new_piece(self) -> None:
        """
        Select a new piece for the game. 
        """
        self.current_piece = random.choice(range(self.nb_pieces))

    def seed(self, seed_value: int) -> None:
        """
        Provide a new seed for the environment.
        """
        random.seed(seed_value)

    def reset(self) -> torch.FloatTensor:
        """
        Reset the game and return the new board state. 

        return: Current board state
        """
        self.board.reset()
        self.new_piece()
        self.lines_cleared = 0
        return self.get_state()

    def render(self):
        print(self.pieces[self.current_piece])
        print(self.board)

    def get_next_states(self) -> dict:
        """
        Returns a dictionary of subsequent states that are reachable with the
        current board state and piece. Used in state-value implementations.

        :return: A dictionary of subsequent states, indexed by the tupel of
            piece orientation and column placement.
        """
        states = {}
        for i in range(self.pieces[self.current_piece].nb_orientations):
            for j in range(self.board_width - self.pieces[self.current_piece].orientations[i].width + 1):
                self.board.drop_piece(
                    self.pieces[self.current_piece].orientations[i], column=j, cancellable=True)
                states[i, j] = self.get_state()
                self.board.cancel_last_move()
        return states

    def step(self, action: Tuple[int, int], new_piece: bool=True) -> Tuple[int, bool]:
        """
        Make one action step given the action. 

        :param action: tuple of piece orientation and column for placement
        :return:
            reward: The number of lines cleared by the step
            done: Boolean indicating if the game is over
        """
        reward = self.board.drop_piece(
            self.pieces[self.current_piece].orientations[action[0]], action[1])
        done = self.board.wall_height > self.board_height
        if new_piece:
            self.new_piece()
        self.lines_cleared += reward
        return reward, done

    def board_step(self) -> Tuple[int, bool]:
        """
        Make one action step using the predetermined weights. Given a board
        state generate all possible successive states. Use the dellacherie
        features and the game feature weights to calculate the state-value of
        the successive states.  Find the maximum state value of a successive
        state and commit the required action to reach it. 
        
        :return: 
            reward: The number of lines cleared by the step
            done: Boolean indicating if the game is over
        """
        actions = np.full((4, self.board_width), -np.inf, dtype='double')
        for i in range(self.pieces[self.current_piece].nb_orientations):
            for j in range(self.board_width - self.pieces[self.current_piece].orientations[i].width + 1):
                self.board.drop_piece(self.pieces[self.current_piece].orientations[i],
                                      column=j, cancellable=True)
                actions[i, j] = (self.get_state()
                                 * self.weights).sum()
                self.board.cancel_last_move()
        a = np.unravel_index(np.argmax(actions), actions.shape)
        reward = self.board.drop_piece(
            self.pieces[self.current_piece].orientations[a[0]], a[1])
        done = self.board.wall_height > self.board_height
        return reward, done

    def play_game(self, render: bool = False) -> Tuple[int, int]:
        """
        Play a game with the current linear weights of the agent. 

        :param render: Indicate if the game should be rendered. 
        """
        cleared = 0
        timestep = 0

        done = False
        while not done:
            reward, done = self.board_step()
            timestep += 1
            cleared += reward
            self.new_piece()
            if render:
                print(self.board)
                print(f"Lines cleared: {cleared:,}")
                print()

        return cleared, timestep


class LinearGameStandard(LinearGame):
    """
    Linear game that utilises a 1D state space for state value approximation.
    """

    def get_state(self) -> torch.FloatTensor:
        """
        Override method that returns the current board as the game state. 

        return: Board state as 1D flattened array.
        """
        return torch.FloatTensor(self.board.board[:self.board_height, :].flatten())


def test_performance(seed: int=12345, nb_games: int=100, log_dir: str='runs', save_dir: str='./'):
    """
    Method to test performance of Dellacherie method.

    :param seed: Seed for the environment 
    :param nb_games: number of episodes to run test for
    :param log_dir: Directory to log TensorBoard results to
    :param save_dir: Directory to save episode reward results to
    """
    runid = time.strftime('%Y%m%dT%H%M%SZ')
    writer = SummaryWriter(log_dir, comment=f"Dellacherie-{runid}")
    lg = LinearGame()
    episode_rewards = []
    episode_duration = []
    for i in range(nb_games):
        reward, timesteps = lg.play_game()
        episode_rewards.append(reward)
        episode_duration.append(timesteps)
        print(f"Episode reward: {reward}, episode duration: {timesteps}")
        lg.reset()
        writer.add_scalar(f"Dellacherie-{runid}/Episode reward", reward, i)
        writer.add_scalar(f"Dellacherie-{runid}/Episode duration", timesteps, i)

    np.array(episode_rewards).tofile(f"{save_dir}/Dellacherie-rewards-{runid}.csv", sep=',')
    np.array(episode_duration).tofile(f"{save_dir}/Dellacherie-timesteps-{runid}.csv", sep=',')
    print(f"Average rewards: {np.mean(np.array(episode_rewards))}")
    print(f"Average duration: {np.mean(np.array(episode_duration))}")


if __name__ == "__main__":
    render = False
    if len(sys.argv) > 1:
        if "render" in sys.argv:
            render = True
        if sys.argv[1] == "test":
            nb_games = 100
            if len(sys.argv) > 2:
                nb_games = int(sys.argv[2])
            test_performance(nb_games=nb_games, save_dir="./runs")
            sys.exit(0)
            
    lg = LinearGame()
    start = time.time()
    print("Starting")
    cleared, duration = lg.play_game(render)
    end = time.time()
    print(f"{cleared} rows cleared")
    print(f"That took {end - start}s")
