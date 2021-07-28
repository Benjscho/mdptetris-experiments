import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Log():
    def __init__(self):
        self.time_d = time.time_ns()
        self.timesteps: int = 0
        self.epochs: int = 0
        self.batch_durations: list[int] = []
        self.episode_rewards: list[list[int]] = []
        self.actor_losses: list[torch.Tensor] = []
        self.avg_ep_rewards = []
        self.avg_ep_durations = []
        self.avg_actor_losses = []
        self.epoch_timesteps = []

    def reset_batches(self):
        self.batch_durations: list[int] = []
        self.episode_rewards: list[list[int]] = []
        self.actor_losses: list[torch.Tensor] = []

    def save(self, save_dir):
        np.array(self.avg_ep_rewards).tofile(f"{save_dir}/avg_ep_rewards.csv", sep=',')
        np.array(self.avg_ep_durations).tofile(f"{save_dir}/avg_ep_durations.csv", sep=',')
        np.array(self.avg_actor_losses).tofile(f"{save_dir}/avg_actor_losses.csv", sep=',')
        np.array(self.epoch_timesteps).tofile(f"{save_dir}/epoch_timesteps.csv", sep=',')


class PPO():
    def __init__(self, args: dict, policy_net, env):

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        # Initialise hyperparams
        self._init_hyperparams(args)

        self.actor = policy_net(self.obs_dim, self.action_dim).to(self.device)
        self.critic = policy_net(self.obs_dim, 1).to(self.device)

        self.optimiser_actor = torch.optim.Adam(
            self.actor.parameters(), lr=self.alpha)
        self.optimiser_critic = torch.optim.Adam(
            self.critic.parameters(), lr=self.alpha)

        self.cov_vars = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_matrix = torch.diag(self.cov_vars)

        self.log = Log()

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

            self.log.timesteps = current_timesteps
            self.log.epochs = epochs

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

                self.log.actor_losses.append(actor_loss.detach())

            self._log()
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
        ep_len_b: list[int] = []

        # Track rewards per episode
        ep_rewards = []

        timesteps = 0
        while timesteps < self.batch_timesteps:
            ep_rewards = []

            obs = self.env.reset().to(self.device)
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
        action_b = torch.tensor(action_b, dtype=torch.float).to(self.device)
        log_probs_b = torch.tensor(log_probs_b, dtype=torch.float).to(self.device)
        rewards_tg_b = self.rewards_to_go(rewards_b)

        self.log.episode_rewards = rewards_b
        self.log.batch_durations = ep_len_b

        return state_b, action_b, log_probs_b, rewards_tg_b, ep_len_b

    def rewards_to_go(self, rewards_b):

        batch_rtg = []

        for ep_reward in reversed(rewards_b):
            discounted_reward = 0
            for reward in reversed(ep_reward):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtg.insert(0, discounted_reward)

        batch_rtg = torch.tensor(batch_rtg, dtype=torch.float).to(self.device)
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

        res = self.actor(torch.FloatTensor(state))

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
        Estimate observation values and log probabilities of actions. 
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
        torch.save(self.actor, f"{self.save_dir}/actor")
        torch.save(self.critic, f"{self.save_dir}/critic")
        self.log.save(self.save_dir)

    def _init_hyperparams(self, args: dict):
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
        self.gpu = 0
        self.save_dir = None
        self.log_dir = None
        self.comment = None
        self.seed = None

        for arg, val in args.items():
            if type(val) == str:
                exec(f'self.{arg} = "{val}"')
            else:
                exec(f'self.{arg} = {val}')

        self.device = torch.device(
            f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")

        # Setup runid, save dir, and tensorboard writer
        self.runid = time.strftime('%Y%m%dT%H%M%SZ')
        self.save_dir = f"{self.save_dir}-{self.runid}"
        self.writer = SummaryWriter(
            self.log_dir, comment=f"{self.comment}-{self.runid}")
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        with open(f"{self.save_dir}/args.txt", 'w') as f:
            f.write(str(args))

        # Set seed
        if self.seed == None:
            seed = int(time.time())
            random.seed(seed)
            self.env.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)
        else:
            random.seed(self.seed)
            self.env.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            else:
                torch.manual_seed(self.seed)

    def _log(self):
        """
        Log info about training to TensorBoard and print to console. 
        """
        rewards_per_timestep = sum(
            [sum(i) for i in self.log.episode_rewards]) / sum(self.log.batch_durations)
        avg_ep_length = np.mean(self.log.batch_durations)
        avg_ep_rewards = np.mean([np.sum(i)
                                 for i in self.log.episode_rewards])
        avg_actor_loss = np.mean([losses.float().mean()
                                 for losses in self.log.actor_losses])

        self.log.avg_ep_rewards.append(avg_ep_rewards)
        self.log.avg_ep_durations.append(avg_ep_length)
        self.log.avg_actor_losses.append(avg_actor_loss)
        self.log.epoch_timesteps.append(self.log.timesteps)
        
        print(flush=True)
        print(
            f"---------------------- Iteration {self.log.epochs} -------------", flush=True)
        print(f"Average episode length: {avg_ep_length}", flush=True)
        print(f"Average episode reward: {avg_ep_rewards}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps so far: {self.log.timesteps}", flush=True)
        print("-------------------------------------------------", flush=True)
        print(flush=True)

        self.writer.add_scalar(
            f'PPO-{self.runid}/Average episode length', avg_ep_length, self.log.epochs)
        self.writer.add_scalar(
            f'PPO-{self.runid}/Average episode rewards', avg_ep_rewards, self.log.epochs)
        self.writer.add_scalar(
            f'PPO-{self.runid}/Average actor loss', avg_actor_loss, self.log.epochs)
        self.writer.add_scalar(f'PPO-{self.runid}/Rewards per 100 timesteps',
                               rewards_per_timestep * 100, self.log.timesteps)

        self.log.reset_batches()
