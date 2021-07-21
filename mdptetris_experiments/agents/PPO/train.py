# Logging
# Training
# Env Setup - multiprocess for batches 
# Gut main, leave parsing, and logging, add in PPO training + models 

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
from gym_mdptetris.envs import board, piece, tetris
from mdptetris_experiments.agents.FFNN import NNHeuristic, NN1D
from mdptetris_experiments.agents.linear_agent import LinearGame, LinearGameStandard, MultiLinearGame
from torch import nn
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument("--init_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--target_network_update", type=int, default=5)
    parser.add_argument("--saving_interval", type=int, default=500)
    parser.add_argument("--epsilon_decay_period", type=int, default=2000)
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

class PPO():
    def __init__(self, args: argparse.Namespace, policy_net, env):
        self.args = args

        self.env = env 
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        # Initialise hyperparams
        self._init_hyperparams(args)
        

        self.actor = policy_net(self.obs_dim, self.action_dim)
        self.critic = policy_net(self.obs_dim, 1)

        self.optimiser_actor = torch.optim.Adam(self.actor.parameters(), lr = args.alpha)
        self.optimiser_critic = torch.optim.Adam(self.critic.parameters(), lr = args.alpha)

        self.cov_vars = torch.full(size=(self.action_dim), fill_value=0.5)
        self.cov_matrix = torch.diag(self.cov_vars)


    def train(self, total_timesteps):
        """
        Train the agent networks for a number of timesteps. 
        """
        pass

    def rollout(self):
        """
        Conduct a rollout 
        """
        state_b = []
        action_b = []
        log_probs_b = []
        rewards_b = []
        rewards_tg_b = []
        ep_len_b = []

        # Track rewards per episode
        ep_rewards = []

        timesteps = 0
        while timesteps < self.batch_timesteps:
            ep_rewards = []

            obs = env.reset()
            done = False

            for ep_t in range(self.max_episode_timesteps):
                timesteps += 1
                state_b.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                ep_rewards.append(rew)
                action_b.append(action)
                log_probs_b.append(log_prob)
                if done:
                    break
            
            rewards_b.append(ep_rewards)
            ep_len_b.append(ep_t + 1)
        
        state_b = torch.tensor(state_b, dtype=torch.float).to(self.device)

    def rewards_to_go(self):
        pass

    def get_action(self, state):
        """
        Query the actor network to get the action.

        :param state: observation at the current timestep

        return:
            action: The action to take
            log_prob: The log probability of the selected action
        """
        # Get mean action
        res = self.actor(state)

        # Create distribution from mean
        dist = torch.distributions.MultivariateNormal(res, self.cov_matrix)

        # Sample action from distribution
        action = dist.sample()

        # Calculate action log probability
        log_prob = dist.log_prob(action)

        # Return sampled action and its log probability
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self):
        """
        Estimate observation values. 
        """
        V = self.critic()
        pass

    def save(self):
        """
        Save current agent state and associated run log details. 
        """
        pass

    def _init_hyperparams(self, args: argparse.Namespace):

        

        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

        for arg, val in vars(args).items():
            exec(f'self.{arg} = {val}')

    def _log(self):
        """
        Log info about training 
        """
