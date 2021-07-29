import os
import random
import time
import argparse
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from mdptetris_experiments.agents.action_networks import PPONN
from torch import nn
import torch.nn.functional as functional
from torch.utils.tensorboard import SummaryWriter
from gym_mdptetris.envs.tetris import TetrisFlat

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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--saving_interval", type=int, default=500)
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

class MultiEnv:
    def __init__(self, board_height, board_width, seed, nb_envs):
        self.agent_con, self.env_con = zip(*[multiprocessing.Pipe() for _ in range(nb_envs)])
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
        np.array(self.avg_ep_rewards).tofile(f"{save_dir}/avg_ep_rewards.csv", sep=',')
        np.array(self.avg_ep_durations).tofile(f"{save_dir}/avg_ep_durations.csv", sep=',')
        np.array(self.avg_actor_losses).tofile(f"{save_dir}/avg_actor_losses.csv", sep=',')
        np.array(self.epoch_timesteps).tofile(f"{save_dir}/epoch_timesteps.csv", sep=',')


class PPO():
    def __init__(self, args: dict):

        self.mp = multiprocessing.get_context("spawn")
        # Initialise hyperparams
        self._init_hyperparams(args)

        self.envs = MultiEnv(board_height=self.board_height,
                    board_width=self.board_width, seed=self.seed, nb_envs=self.nb_games)
        print("Agents created")
        self.model = PPONN(self.envs.observation_space, self.envs.action_space).to(self.device)
        self.model.share_memory()

        self.cov_vars = torch.full(size=(self.envs.action_space,), fill_value=0.5).to(self.device)
        self.cov_matrix = torch.diag(self.cov_vars).to(self.device)

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        
        # Initialise log
        self.log = Log()

    def train(self):
        """
        Train the agent networks for a number of timesteps. 
        """
        print("training")
        [connection.send(("reset", None)) for connection in self.envs.agent_con]
        obs = [connection.recv() for connection in self.envs.agent_con]
        obs = torch.FloatTensor(obs).to(self.device)

        epoch = 0
        timesteps = 0
        while True:
            epoch += 1
            old_log_pols = []
            actions = []
            values = []
            states = []
            rewards = []
            dones = []
            for _ in range(self.max_episode_timesteps):
                timesteps += 1 * self.nb_games
                states.append(obs)
                logits, value = self.model(obs)

                dist = torch.distributions.MultivariateNormal(logits, self.cov_matrix)

                # Sample action from distribution
                action = dist.sample()

                # Calculate action log probability
                old_log_policy = dist.log_prob(action)

                values.append(value)
                #policy = functional.softmax(logits, dim=1)
                #old_m = torch.distributions.Categorical(policy)
                #action = old_m.sample()
                actions.append(action)
                #old_log_policy = old_m.log_prob(action)
                old_log_pols.append(old_log_policy)
                for conn, action in zip(self.envs.agent_con, action.cpu().numpy()):
                    conn.send(("step", action))
                
                obs, reward, done, info = zip(*[connection.recv() for connection in self.envs.agent_con])
                obs = torch.FloatTensor(obs).to(self.device)
                reward = torch.FloatTensor(reward).to(self.device)
                done = torch.FloatTensor(done).to(self.device)
                rewards.append(reward)
                dones.append(done)
                if torch.any(done): 
                    [connection.send(("reset", None)) for connection in self.envs.agent_con]
                    obs = [connection.recv() for connection in self.envs.agent_con]
                    obs = torch.FloatTensor(obs).to(self.device)
            
            _, new_value = self.model(obs)
            new_value = new_value.squeeze()
            old_log_pols = torch.cat(old_log_pols).detach()
            actions = torch.cat(actions)
            values = torch.cat(values).detach().squeeze()
            states = torch.cat(states)

            gae = 0
            R = []
            for value, reward, done in list(zip(values, rewards, dones))[::-1]:
                gae = gae * self.gamma 
                gae = gae + reward + (0 if done else self.gamma * new_value.detach()) - value.detach()
                new_value = value
                R.append(gae + value)
            R = R[::-1]
            R = torch.cat(R).detach()
            advantages = R - values
            print("update iters")
            for i in range(self.updates_per_iter):
                ind = torch.randperm(self.max_episode_timesteps * self.nb_games)
                for j in range(self.batch_size):
                    batch_indices = ind[int(j * (self.max_episode_timesteps * self.nb_games / self.batch_size)): int(
                        (j+1)*(self.max_episode_timesteps * self.nb_games / self.batch_size))]
                    policy, value = self.model(states[batch_indices])
                    new_pol = functional.softmax(policy)

                    dist = torch.distributions.MultivariateNormal(new_pol, self.cov_matrix)

                    # Calculate action log probability
                    new_log_policy = dist.log_prob(actions[batch_indices])

                    ratio = torch.exp(new_log_policy - old_log_pols[batch_indices])
                    actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices], torch.clamp(
                        ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages[batch_indices]))
                    critic_loss = functional.smooth_l1_loss(R[batch_indices], value.squeeze())
                    entropy_loss = torch.mean(dist.entropy())
                    total_loss = actor_loss + critic_loss - self.beta * entropy_loss
                    self.optimiser.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimiser.step()
            print(f"Epoch: {epoch}, Total loss: {total_loss}")
            print(f"Timesteps: {timesteps} Average rewards: {np.mean(rewards)}")

    def evaluate(self):
        """
        Evaluate current model and watch progress while model trains. 
        """
        env = TetrisFlat(board_height=self.board_height,
                     board_width=self.board_width, seed=self.seed)
        
        model = PPONN(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
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

    def save(self):
        """
        Save current agent state and associated run log details. 
        """
        torch.save(self.model, f"{self.save_dir}/model")
        self.log.save(self.save_dir)

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
        self.gamma = 0.99
        self.beta = 0.01
        self.saving_interval = 50
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
        

    def _log(self):
        """
        Log info about training to TensorBoard and print to console. 
        """
        rewards_per_timestep = sum(
            [sum(i) for i in self.log.episode_rewards]) / sum(self.log.batch_durations)
        avg_ep_length = np.mean(self.log.batch_durations)
        avg_ep_rewards = np.mean([np.sum(i)
                                 for i in self.log.episode_rewards])
        avg_actor_loss = np.mean([losses.cpu().float().mean()
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

if __name__=="__main__":
    args = vars(get_args())

    agent = PPO(args)
    agent.train()