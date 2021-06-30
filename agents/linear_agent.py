import os
import random
import time
import numpy as np
import gym_mdptetris.envs
from gym_mdptetris.envs import board, piece, feature_functions

class LinearGame():
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
        res = np.empty((6), dtype='double')
        for i, f in enumerate(feature_functions.get_dellacherie_funcs()):
            res[i] = f(self.board)
        return res 
    
    def new_piece(self):
        self.current_piece = random.choice(range(self.nb_pieces))
    
    def seed(self, seed_value: int):
        random.seed(seed_value)

    def board_step(self):
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