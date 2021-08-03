import argparse
import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import torch.nn.functional as functional
from gym_mdptetris.envs.tetris import TetrisFlat
from mdptetris_experiments.agents.action_networks import PPONN, NN1DAction
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0',
                        help="The GPU to train the agent on")
    parser.add_argument("--board_height", type=int, default=20)
    parser.add_argument("--board_width", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--batch_timesteps", type=int, default=10000)
    parser.add_argument("--max_episode_timesteps", type=int, default=2000)
    parser.add_argument("--nb_games", type=int, default=8)
    parser.add_argument("--updates_per_iter", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--saving_interval", type=int, default=500)
    parser.add_argument("--state_rep", type=str, default="1D")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--load_dir", type=str, default=None,
                        help="Path to partially trained actor and critic models.")
    parser.add_argument("--save_dir", type=str,
                        default=f"runs/run-info")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--comment", type=str, default="test",
                        help="Run comment for TensorBoard writer.")

    args = parser.parse_args()
    return args


class MultiEnv:
    def __init__(self, board_height, board_width, seed, nb_envs):
        self.agent_con, self.env_con = zip(
            *[multiprocessing.Pipe() for _ in range(nb_envs)])
        self.nb_envs = nb_envs
        self.envs = []
        for _ in range(nb_envs):
            self.envs.append(TetrisFlat(board_height=board_height,
                                        board_width=board_width, seed=seed))
        self.observation_space = self.envs[0].observation_space.shape[0]
        self.action_space = self.envs[0].action_space.shape[0]
        for i in range(nb_envs):
            process = multiprocessing.Process(target=self.run_env, args=(i,))
            process.start()
            self.env_con[i].close()

    def run_env(self, index):
        self.agent_con[index].close()
        while True:
            request, action = self.env_con[index].recv()
            if request == 'step':
                self.env_con[index].send(self.envs[index].step(action))
            elif request == 'reset':
                self.env_con[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError


class Log():
    def __init__(self):
        self.time_d = time.time_ns()
        self.timesteps: int = 0
        self.epochs: int = 0
        self.batch_durations: list[int] = []
        self.episode_rewards: list[list[int]] = []
        self.actor_losses = []
        self.avg_ep_rewards = []
        self.avg_ep_durations = []
        self.avg_actor_losses = []
        self.epoch_timesteps = []

    def reset_batches(self):
        self.batch_durations: list[int] = []
        self.episode_rewards: list[list[int]] = []
        self.actor_losses = []

    def save(self, save_dir):
        np.array(self.avg_ep_rewards).tofile(
            f"{save_dir}/avg_ep_rewards.csv", sep=',')
        np.array(self.avg_ep_durations).tofile(
            f"{save_dir}/avg_ep_durations.csv", sep=',')
        np.array(self.avg_actor_losses).tofile(
            f"{save_dir}/avg_actor_losses.csv", sep=',')
        np.array(self.epoch_timesteps).tofile(
            f"{save_dir}/epoch_timesteps.csv", sep=',')


class PPO():
    def __init__(self, args: dict):

        # Initialise hyperparams
        self._init_hyperparams(args)

        self.envs = MultiEnv(board_height=self.board_height,
                             board_width=self.board_width, seed=self.seed, nb_envs=self.nb_games)
        print("Agents created")
        self.actor = NN1DAction(
            self.envs.observation_space, self.envs.action_space).to(self.device)
        self.actor.share_memory()
        self.critic = NN1DAction(
            self.envs.observation_space, 1).to(self.device)
        self.critic.share_memory()

        self.cov_vars = torch.full(
            size=(self.envs.action_space,), fill_value=0.5).to(self.device)
        self.cov_matrix = torch.diag(self.cov_vars).to(self.device)

        self.actor_optimiser = torch.optim.Adam(
            self.actor.parameters(), lr=self.alpha)
        self.critic_optimiser = torch.optim.Adam(
            self.critic.parameters(), lr=self.alpha)

        # Initialise log
        self.log = Log()

    def train(self):
        """
        Train the agent networks for a number of timesteps. 
        """
        print("training")
        [connection.send(("reset", None))
         for connection in self.envs.agent_con]
        obs = [connection.recv() for connection in self.envs.agent_con]
        obs = torch.FloatTensor(obs).to(self.device)

        epoch = 0
        timesteps = 0
        while True:
            epoch += 1
            log_probs = []
            actions = []
            states = []
            rewards = []
            dones = []
            for _ in range(self.max_episode_timesteps):
                timesteps += 1 * self.nb_games
                states.append(obs)
                logits = self.actor(obs)

                dist = torch.distributions.MultivariateNormal(
                    logits, self.cov_matrix)

                # Sample action from distribution
                action = dist.sample()

                # Calculate action log probability
                log_prob = dist.log_prob(action)

                actions.append(action)
                log_probs.append(log_prob)

                for conn, action in zip(self.envs.agent_con, action.cpu().numpy()):
                    conn.send(("step", action))

                obs, reward, done, info = zip(
                    *[connection.recv() for connection in self.envs.agent_con])
                obs = torch.FloatTensor(obs).to(self.device)
                reward = torch.FloatTensor(reward).to(self.device)
                done = torch.BoolTensor(done).to(self.device)
                rewards.append(reward)
                dones.append(done)
                for i in range(self.nb_games):
                    if done[i]:
                       self.envs.agent_con[i].send(("reset", None))
                       obs[i] = torch.FloatTensor(
                           self.envs.agent_con[i].recv())

            #self.log.episode_rewards = rewards

            log_probs = torch.cat(log_probs).detach()
            actions = torch.cat(actions)
            states = torch.cat(states)
            rewards_tg = self.rewards_to_go(rewards, dones).to(self.device)

            # Calculate advantage estimates
            V, _ = self.evaluate(states, actions)
            A_k = rewards_tg - V.detach()

            # Normalise advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.updates_per_iter):
                V, curr_log_probs = self.evaluate(states, actions)

                ratios = torch.exp(curr_log_probs - log_probs)

                # Calculate surrogate losses 
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, rewards_tg)

                self.actor_optimiser.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimiser.step()

                self.critic_optimiser.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimiser.step()

                #self.log.actor_losses.append(actor_loss.detach())

            avg_rewards = torch.mean(torch.cat(rewards))
            print(f"Epoch: {epoch}, Actor loss: {actor_loss}")
            print(f"Timesteps: {timesteps} Average rewards: {avg_rewards}")
            self.writer.add_scalar(
                f'PPO-{self.runid}/Average episode reward', avg_rewards, timesteps)
            self.writer.add_scalar(
                f'PPO-{self.runid}/Total loss', actor_loss, epoch)

            if epoch % self.saving_interval:
                self.save()

    def rewards_to_go(self, rewards, done) -> torch.Tensor:
        """
        Calculate rewards to go.

        :return: Batch of rewards to go 
        """
        rtg_batch = []
        discounted_reward = torch.zeros(rewards[0].size()).to(self.device)
        for reward, done in zip(reversed(rewards), reversed(done)):
            discounted_reward = reward + discounted_reward * self.alpha
            rtg_batch.insert(0, discounted_reward)
            discounted_reward = discounted_reward * ~done
        return torch.cat(rtg_batch)

    def run_demo(self):
        """
        #TODO: Evaluate current model and watch progress while model trains. 
        """
        env = TetrisFlat(board_height=self.board_height,
                         board_width=self.board_width, seed=self.seed)

        model = PPONN(env.observation_space.shape[0], env.action_space.shape[0]).to(
            self.device)
        model.eval()
        obs = env.reset()
        timestep = 0
        done = True
        while True:
            timestep += 1
            if done:
                model.load_state_dict(self.model.state_dict())

            obs = torch.FloatTensor(obs).to(self.device)
            probs, value = model(obs)
            distr = functional.softmax(probs)
            action = torch.argmax(distr).item()
            obs, reward, done, info = env.step(action)

            if timestep > self.total_training_steps:
                done = True
            if done:
                timestep = 0
                obs = env.reset()

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
        torch.save(self.actor.state_dict(), f"{self.save_dir}/actor.pt")
        torch.save(self.critic.state_dict(), f"{self.save_dir}/critic.pt")
        self.log.save(self.save_dir)

    def load(self):
        """
        Load pre-trained models from save files. 
        """
        if self.load_dir == None:
            raise ValueError("No load file given")
        actor = f"{self.load_dir}/actor.pt"
        critic = f"{self.load_dir}/critic.pt"
        if not os.path.exists(actor):
            raise ValueError("Actor model does not exist")
        if not os.path.exists(critic):
            raise ValueError("Critic model does not exist")
        
        self.actor.load_state_dict(torch.load(actor)).to(self.device)
        self.actor.eval()
        self.critic.load_state_dict(torch.load(critic)).to(self.device)
        self.critic.eval()

    def test(self):
        pass

    def _init_hyperparams(self, args: dict):
        # Set default hyperparams
        self.board_height = 20
        self.board_width = 10
        self.batch_size = 16
        self.batch_timesteps = 10000
        self.max_episode_timesteps = 512
        self.total_training_steps = 2e7
        self.nb_games = 8
        self.alpha = 1e-3
        self.beta = 0.01
        self.saving_interval = 50
        self.clip = 0.2
        self.updates_per_iter = 5
        self.gpu = 0
        self.save_dir = None
        self.log_dir = None
        self.load_dir = None
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
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)
        else:
            random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            else:
                torch.manual_seed(self.seed)

    def _log(self, total_loss, epoch, rewards, timesteps):
        """
        #TODO Log info about training to TensorBoard and print to console. 
        """

def train(args: dict):
    agent = PPO(args)
    agent.train()

def test(args: dict):
    agent = PPO(args)
    agent.load()
    agent.test()

if __name__ == "__main__":
    args = vars(get_args())

    agent = PPO(args)
    agent.train()
