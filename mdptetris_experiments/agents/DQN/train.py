import os
import torch
import gym
import argparse

import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter 

import gym_mdptetris
from gym_mdptetris.envs import board, piece, tetris
from mdptetris_experiments.agents.DQN.DQ_network import DQ_network
from mdptetris_experiments.agents.linear_agent import LinearGame

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_buffer", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--nb_epochs", type=int, default=5000)
    parser.add_argument("--epsilon_decay_period", type=int, default=3000)
    parser.add_argument("--board_height", type=int, default=20)
    parser.add_argument("--board_width", type=int, default=10)
    parser.add_argument("--state_representation", type=str, default="board-2D")
    parser.add_argument("--log_dir", type=str, default="tensorboard")

    args = parser.parse_args()
    return args


def train(args: argparse.Namespace):
    # Set up environment
    env = LinearGame()

    writer = SummaryWriter()
    # Set up model - network + optimizer 
    model = DQ_network()
    # 

if __name__=='__main__':
    # Train
    args = get_args()
    train(args) 