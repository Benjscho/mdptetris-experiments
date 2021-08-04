import os
import random
import time

import gym_mdptetris.envs
import numpy as np
from gym_mdptetris.envs import board, piece


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
        while self.board.wall_height < self.board_height:
            cleared += self.board_step()
            self.new_piece()
            if render:
                print(self.board)

        return cleared


def test_performance(seed: int):
    """
    Method to test random agent performance. 

    :param seed: Seed value for environment. 
    """
    pass


if __name__ == "__main__":
    lg = RandomLinearGame()
    start = time.time()
    lg.play_game()
    end = time.time()
    print(f"That took {end - start}s")
