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
        current_timesteps = 0
        epochs = 0
        while current_timesteps < total_timesteps:
            state_b, action_b, log_probs_b, rewards_tg_b, ep_len_b = self.rollout()
            current_timesteps += np.sum(ep_len_b)

            epochs += 1

            # Calculate advantage for current iteration
            V, _ = self.evaluate(state_b, action_b)
            A_k = rewards_tg_b - V.detach()

            # Normalise advantages to decrease variance and improve convergence
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.updates_per_iter):
                V, curr_log_probs = self.evaluate(state_b, action_b)

                # Calculate ratio
                ratios = torch.exp(curr_log_probs - log_probs_b)

                # Surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, rewards_tg_b)

                self.optimiser_actor.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.optimiser_actor.step()

                self.optimiser_critic.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.optimiser_critic.step()

            if epochs % self.saving_interval == 0:
                self.save() 

    def rollout(self):
        """
        Conduct a rollout 
        """
        state_b = []
        action_b = []
        log_probs_b = []
        rewards_b = []
        ep_len_b = []

        # Track rewards per episode
        ep_rewards = []

        timesteps = 0
        while timesteps < self.batch_timesteps:
            ep_rewards = []

            obs = self.env.reset()
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
        
        state_b = torch.tensor(state_b, dtype=torch.float)
        action_b = torch.tensor(action_b, dtype=torch.float)
        log_probs_b = torch.tensor(log_probs_b, dtype=torch.float)
        rewards_tg_b = self.rewards_to_go(rewards_b)

        return state_b, action_b, log_probs_b, rewards_tg_b, ep_len_b

    def rewards_to_go(self, rewards_b):

        batch_rtg = []

        for ep_reward in reversed(rewards_b):
            discounted_reward = 0
            for reward in reversed(ep_reward):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtg.insert(0, discounted_reward)

        batch_rtg = torch.tensor(batch_rtg, dtype=torch.float)
        return batch_rtg

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

    def evaluate(self, state_b, action_b):
        """
        Estimate observation values. 
        """
        V = self.critic(state_b).squeeze()

        res = self.actor(state_b)
        dist = torch.distributions.MultivariateNormal(res, self.cov_matrix)
        log_probs = dist.log_prob(action_b)
        return V, log_probs

    def save(self):
        """
        Save current agent state and associated run log details. 
        """
        pass

    def _init_hyperparams(self, args: argparse.Namespace):
        # Set default hyperparams
        self.board_height = 20
        self.board_width = 10
        self.batch_size = 512
        self.batch_timesteps = 10000
        self.max_episode_timesteps = 2000
        self.nb_games = 20
        self.alpha = 1e-3
        self.gamma = 0.99
        self.saving_interval = 500
        self.clip = 0.2
        self.updates_per_iter = 5

        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

        for arg, val in vars(args).items():
            exec(f'self.{arg} = {val}')

        # Setup runid, save dir, and tensorboard writer
        runid = time.strftime('%Y%m%dT%H%M%SZ')
        save_dir = f"{args.save_dir}-{runid}"
        writer = SummaryWriter(args.log_dir, comment=f"{args.comment}-{runid}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(f"{save_dir}/args.txt", 'w') as f:
            f.write(str(args))

        # Set seed 
        if args.seed == None:
            seed = int(time.time())
            random.seed(seed)
            self.env.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)
        else:
            random.seed(args.seed)
            self.env.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
            else:
                torch.manual_seed(args.seed)

    def _log(self):
        """
        Log info about training to TensorBoard and print to console. 
        """
