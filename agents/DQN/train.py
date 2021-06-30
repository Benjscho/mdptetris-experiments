import os
import torch
import gym
import argparse

import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter 

import gym_mdptetris
from gym_mdptetris import board, piece
from DQ_network import DQ_network

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_buffer", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--nb_epochs", type=int, default=5000)
    parser.add_argument("--epsilon_decay_period", type=int, default=3000)


    args = parser.parse_args()
    return args


def train(args: argparse.Namespace):
    # Set up environment

    env = gym.make("gym_mdptetris:melaxtetris-v0")

    writer = SummaryWriter()
    # Set up model - network + optimizer 
    model = DQ_network() 

    # 

if __name__=='__main__':
    # Train
    args = get_args()
    train(args) 