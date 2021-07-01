import os
import random
import time
import numpy as np
import gym_mdptetris.envs
from gym_mdptetris.envs import board, piece, feature_functions

class LinearGame():
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
    def __init__(self, weights=np.array([-1, 1, -1, -1, -4, -1]), board_height=20, board_width=10,  piece_set='pieces4.dat', seed=12345):
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
        self.board = board.Board(max_piece_height=self.max_piece_height,
            width=board_width, height=board_height)

    def get_dellacherie_features(self) -> np.ndarray:
        """
        Get the heuristic representation of a board state as the set of
        6 Dellacherie features.

        :return: Numpy array of the Dellacherie features of the current
            board state.
        """
        res = np.empty((6), dtype='double')
        for i, f in enumerate(feature_functions.get_dellacherie_funcs()):
            res[i] = f(self.board)
        return res 
    
    def new_piece(self):
        """
        Select a new piece. 
        """
        self.current_piece = random.choice(range(self.nb_pieces))
    
    def seed(self, seed_value: int):
        """
        Provide a new seed for the environment.
        """
        random.seed(seed_value)

    def get_next_states(self):
        """
        Returns a dictionary of subsequent states that are reachable with the
        current board state and piece. Used in state-value implementations.

        :return: A dictionary of subsequent states, indexed by the tupel of
            piece orientation and column placement.
        """
        states = {} 
        for i in range(self.pieces[self.current_piece].nb_orientations):
            for j in range(self.board_width - self.pieces[self.current_piece].orientations[i].width + 1):
                self.board.drop_piece(self.pieces[self.current_piece].orientations[i], column=j, cancellable=True)
                states[i,j] = self.get_dellacherie_features()
                self.board.cancel_last_move()
        return states

    def board_step(self):
        """
        Make one action step. Given a board state generate all possible 
        successive states. Use the dellacherie features and the game feature 
        weights to calculate the state-value of the successive states. 
        Find the maximum state value of a successive state and commit the
        required action to reach it. 

        :return: Returns the number of lines cleared by the step
        """
        actions = np.full((4, self.board_width), -np.inf, dtype='double')
        for i in range(self.pieces[self.current_piece].nb_orientations):
            for j in range(self.board_width - self.pieces[self.current_piece].orientations[i].width + 1):
                self.board.drop_piece(self.pieces[self.current_piece].orientations[i],
                    column=j, cancellable=True)
                actions[i,j] = (self.get_dellacherie_features() * self.weights).sum()
                self.board.cancel_last_move()
        a = np.unravel_index(np.argmax(actions), actions.shape)
        return self.board.drop_piece(self.pieces[self.current_piece].orientations[a[0]], a[1])

    def play_game(self, render=False):
        cleared = 0
        while self.board.wall_height < self.board_height and cleared < 1e4:
            cleared += self.board_step()
            self.new_piece()
            if render:
                print(self.board)
        
        print(f"{cleared} rows cleared")

if __name__=="__main__":
    lg = LinearGame()
    start = time.time()
    print("Starting")
    lg.play_game()
    end = time.time()
    print(f"That took {end - start}s")