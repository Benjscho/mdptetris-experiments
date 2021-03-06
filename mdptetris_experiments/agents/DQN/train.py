import argparse

from torch.nn import functional
from mdptetris_experiments.agents.action_networks import NN1DAction
import os
import random
import time
from collections import deque

import numpy as np
import torch
from gym_mdptetris.envs import board, piece, tetris
from gym_mdptetris.envs.tetris import TetrisFlat
from mdptetris_experiments.agents.FFNN import NN1D, NNHeuristic
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--board_height", type=int, default=20)
    parser.add_argument("--board_width", type=int, default=10)
    parser.add_argument("--replay_buffer_length", type=int, default=20000)
    parser.add_argument("--training_start", type=int, default=2000,
                        help="Minimum timesteps for training to start.")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1e-4,
                        help="Optimiser learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--total_timesteps", type=int, default=1e7)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--target_network_update", type=int, default=5,
                        help="Epoch interval to update the target network.")
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



class DQN:
    def __init__(self, args: argparse.Namespace):
        """
        Class that implements a model-based DQN agent to learn a game of Tetris.
        The model for the environment is provided in the linear_agent file, 
        which allows generation of subsequent states, and retrieval of their
        representation as either the full board, or as a set of features. 

        :param args: A Namespace object containing experiment hyperparameters
        """
        self.env = TetrisFlat(board_height=args.board_height,
                              board_width=args.board_width, seed=args.seed)

        self._init_hyperparams(args)

        input_dims = self.env.observation_space.shape[0]
        output_dims = self.env.action_space.shape[0]
        # Initialise models
        self.model = NN1DAction(input_dims, output_dims).to(self.device)
        self.target = NN1DAction(input_dims, output_dims).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.alpha)
        self.replay_buffer = deque(maxlen=args.replay_buffer_length)
        self.loss_criterion = nn.MSELoss()

    def train(self):
        """
        Method to train the agent. Iterates through timesteps to gather training
        data, which is then stored in the buffer. After an episode concludes, makes
        a training step. Outputs information on the current training status
        of the agent while training, and saves the trained model at intervals. 
        """
        self.epochs = []
        self.timesteps = []
        state = self.env.reset().to(self.device)

        self.epoch = 0
        self.timestep = 0
        ep_score = 0
        while self.epoch < self.total_epochs:
            action = self.get_action(state)

            new_state, reward, done, info = self.env.step(action)

            ep_score += reward
            self.timestep += 1

            self.replay_buffer.append([state, action, reward, new_state, done])
            self.timesteps.append(reward)
            if done:
                self.update_model()
                if self.epoch > 0:
                    self._log(ep_score)
                ep_score = 0
                state = self.env.reset().to(self.device)
            else:
                state = new_state.to(self.device)

    def test(self, nb_episodes: int=1000):
        """
        Method to test the performance of a trained agent for specified
        number of episodes. Outputs performance during testing and saves
        results to csv files. The agent is loaded from the pre-specified
        load file passed when the agent is instantiated. 

        :param nb_episodes: Number of episodes to test the trained agent for.
        """
        self.load()
        episode_rewards = []
        episode_durations = []
        done = False
        self.epsilon = 0
        for i in range(nb_episodes):
            state = self.env.reset()
            ep_score = 0
            timesteps = 0
            while not done:
                action, _ = self.get_action_and_new_state()
                reward, done = self.env.step(action)
                ep_score += reward
                timesteps += 1

            episode_rewards.append(ep_score)
            episode_durations.append(timesteps)
            print(f"Episode reward: {ep_score}, episode duration: {timesteps}")
            self.writer.add_scalar(f"DQN-{self.runid}/Episode reward", ep_score, i)
            self.writer.add_scalar(f"DQN-{self.runid}/Episode duration", timesteps, i)

        np.array(episode_rewards).tofile(f"{self.save_dir}/DQN-test-rewards-{self.runid}.csv", sep=',')
        np.array(episode_durations).tofile(f"{self.save_dir}/DQN-test-durations-{self.runid}.csv", sep=',')
        print(f"Average rewards: {np.mean(np.array(episode_rewards))}")
        print(f"Average duration: {np.mean(np.array(episode_durations))}")

    def update_model(self):
        """
        Method to perform one update step on the agent model from the state 
        transitions saved in the agent memory. 
        """
        if len(self.replay_buffer) < self.training_start:
            return

        # Increment epoch and decrement epsilon
        self.epoch += 1
        self.epsilon -= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.final_epsilon)

        batch = random.sample(self.replay_buffer, min(
            len(self.replay_buffer), self.batch_size))

        state_b, action_b, reward_b, new_state_b, done_b = zip(*batch)
        state_b = torch.stack(state_b).to(self.device)
        reward_b = torch.from_numpy(
            np.array(reward_b, dtype=np.float32)[:, None]).to(self.device)
        new_state_b = torch.stack(new_state_b).to(self.device)

        # Use model to judge state values, train prediction against target network
        q_vals = self.model(state_b).to(self.device)
        with torch.no_grad():
            next_predictions = self.target(new_state_b)

        y_b = []
        for reward, done, prediction in zip(reward_b, done_b, next_predictions):
            y_b.append(reward if done else reward + self.gamma*prediction)
        y_b = torch.cat(y_b).to(self.device)

        # Calculate loss and train network
        self.optimizer.zero_grad()
        loss = self.loss_criterion(q_vals, y_b)
        loss.backward()
        self.optimizer.step()

        # Update the target network
        if self.epoch % self.target_network_update == 0:
            self.target.load_state_dict(self.model.state_dict())

        if self.epoch % self.saving_interval == 0:
            self.save()

    def get_action(self, state: torch.Tensor):
        """
        Get action. 

        :param state: Current state.
        """
        probs = self.model(state)
        dist = functional.softmax(probs)
        action = torch.argmax(dist).item()
        return action

    def load(self):
        """
        Load trained or partially trained model from load file specified
        in agent parameters. 
        """
        if self.load_file == None:
            raise ValueError("No load file given")

        if self.load_file[:-3] != ".pt":
            self.model = torch.load(self.load_file).to(self.device)
        else:
            self.model.load_state_dict(
                torch.load(self.load_file)).to(self.device)

        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

    def _log(self, ep_score: int):
        """
        Log information about the current epoch to output and TensorBoard.

        :param ep_score: score from the previous episode.
        """
        self.epochs.append(ep_score)
        print(f"Epoch: {self.epoch}, score: {ep_score}")
        self.writer.add_scalar(f'Train-{self.runid}/Lines cleared per epoch',
                               ep_score, self.epoch - 1)
        self.writer.add_scalar(f'Train-{self.runid}/Lines cleared over last 100 timesteps',
                               sum(self.timesteps[-100:]), self.timestep - 1)
        self.writer.add_scalar(
            f'Train-{self.runid}/Epsilon vlaue', self.epsilon, self.epoch - 1)

    def _init_hyperparams(self, args: argparse.Namespace):
        """
        Initialise agent hyperparameters from the arguments passed in.

        :param args: Namespace containing hyperparameters. 
        """
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

        self.runid = time.strftime('%Y%m%dT%H%M%SZ')
        self.save_dir = f"{args.save_dir}-{self.runid}"

        # Writer for TensorBoard
        self.writer = SummaryWriter(
            args.log_dir, comment=f"{args.comment}-{self.runid}")
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        with open(f"{self.save_dir}/args.txt", 'w') as f:
            f.write(str(args))

        self.epsilon = args.init_epsilon
        self.epsilon_decay_rate = (
            args.init_epsilon - args.final_epsilon) / args.epsilon_decay_period
        self.total_epochs = args.epochs
        self.training_start = args.training_start
        self.final_epsilon = args.final_epsilon
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.target_network_update = args.target_network_update
        self.saving_interval = args.saving_interval
        self.load_file = args.load_file

        # Seed randomness
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

    def save(self):
        """
        Method to save the current model and information about the run to disk. 
        """
        torch.save(self.model.state_dict(), f"{self.save_dir}/model.pt")
        np.array(self.epochs).tofile(f"{self.save_dir}/epochs.csv", sep=',')
        np.array(self.timesteps).tofile(
            f"{self.save_dir}/timesteps.csv", sep=',')



if __name__ == '__main__':
    # Train the model
    args = get_args()

    if args.test:
        assert args.load_file != None
        agent = DQN(args)
        agent.test()
    else:
        agent = DQN(args)
        agent.train()
