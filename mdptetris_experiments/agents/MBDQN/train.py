import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
from gym_mdptetris.envs import board, piece, tetris
from mdptetris_experiments.agents.FFNN import (NN1D, NNHeuristic,
                                               NNHeuristicSimple)
from mdptetris_experiments.agents.linear_agent import (LinearGame,
                                                       LinearGameStandard)
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0',
                        help="Select the GPU to run the model on.")
    parser.add_argument("--test", action='store_true',
                        help="Test a trained model")
    parser.add_argument("--render", action='store_true',
                        help="Render the environment")
    parser.add_argument("--one_piece", action='store_true',
                        help="Only train or test model on one piece per episode.")
    parser.add_argument("--board_height", type=int,
                        default=20, help="Height for the Tetris board")
    parser.add_argument("--board_width", type=int, default=10,
                        help="Width for the Tetris board")
    parser.add_argument("--replay_buffer_length", type=int, default=20000,
                        help="Number of timesteps to store in the replay memory buffer")
    parser.add_argument("--training_start", type=int, default=2000,
                        help="Minimum timesteps for training to start.")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Timestep batch size for training the model.")
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="Adam optimiser learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Future reward discount rate")
    parser.add_argument("--init_epsilon", type=float, default=1,
                        help="Initial epsilon value for random action selection.")
    parser.add_argument("--final_epsilon", type=float, default=1e-3,
                        help="Minimum epsilon value for exploration.")
    parser.add_argument("--epochs", type=int, default=3000,
                        help="Number of epochs to train the agent.")
    parser.add_argument("--target_network_update", type=int, default=5,
                        help="Epoch interval to update the target network.")
    parser.add_argument("--saving_interval", type=int, default=500,
                        help="Epoch interval between model checkpoints.")
    parser.add_argument("--epsilon_decay_period", type=int, default=2000,
                        help="Number of epochs to linearly decay the epsilon over.")
    parser.add_argument("--state_rep", type=str, default="heuristic",
                        help="State representation for the Tetris game. Heuristic or 1D.")
    parser.add_argument("--log_dir", type=str, default="runs",
                        help="Directory to save TensorBoard data to.")
    parser.add_argument("--load_file", type=str, default=None,
                        help="Path to partially trained model")
    parser.add_argument("--save_dir", type=str, default=f"runs/run-info",
                        help="Directory to save model and run info to")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed value for environment.")
    parser.add_argument("--comment", type=str, default=None,
                        help="Run comment for TensorBoard writer.")

    args = parser.parse_args()
    return args


# Define state representations and respective NN architectures
state_rep = {
    "heuristic": [NNHeuristic, LinearGame],
    "heuristic-simplenet": [NNHeuristicSimple, LinearGame],
    "1D": [NN1D, LinearGameStandard]
}


class MBDQN:
    def __init__(self, args: argparse.Namespace):
        """
        Class that implements a model-based DQN agent (MBDQN) to learn a game of
        Tetris.  The model for the environment is provided in the linear_agent
        file, which allows generation of subsequent states, and retrieval of
        their representation as either the full board, or as a set of features. 

        Attribution: Approach inspired by Viet Nguyen's: https://github.com/uvipen/Tetris-deep-Q-learning-pytorch
        
        :param args: A Namespace object containing experiment hyperparameters
        """
        self.env = state_rep[args.state_rep][1](board_height=args.board_height,
                                                board_width=args.board_width)

        self._init_hyperparams(args)

        # Initialise models
        input_dims = args.board_height * args.board_width if args.state_rep == "1D" else 6
        self.model = state_rep[args.state_rep][0](input_dims).to(self.device)
        self.target = state_rep[args.state_rep][0](input_dims).to(self.device)
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
            action, new_state = self.get_action_and_new_state()

            if self.render:
                self.env.render()
            reward, done = self.env.step(action, self.one_piece)
            ep_score += reward
            self.timestep += 1

            self.replay_buffer.append([state, reward, new_state, done])
            self.timesteps.append(reward)

            # Train the model if the episode has concluded and update log
            if done:
                self.update_model()
                if self.epoch > 0:
                    self._log(ep_score)
                ep_score = 0
                state = self.env.reset().to(self.device)
            else:
                state = new_state.to(self.device)

    def test(self, nb_episodes: int = 1000):
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
        self.epsilon = 0
        print("Testing start:")
        for i in range(nb_episodes):
            done = False
            state = self.env.reset()
            ep_score = 0
            timesteps = 0
            while not done:
                if self.render:
                    self.env.render()
                action, _ = self.get_action_and_new_state()
                reward, done = self.env.step(action, self.one_piece)
                ep_score += reward
                timesteps += 1
                if not self.one_piece and ep_score > 100:
                    break

            episode_rewards.append(ep_score)
            episode_durations.append(timesteps)
            print(
                f"Episode: {i}, Episode reward: {ep_score}, Episode duration: {timesteps}")
            self.writer.add_scalar(
                f"DQN-{self.runid}/Episode reward", ep_score, i)
            self.writer.add_scalar(
                f"DQN-{self.runid}/Episode duration", timesteps, i)

        np.array(episode_rewards).tofile(
            f"{self.save_dir}/DQN-test-rewards-{self.runid}.csv", sep=',')
        np.array(episode_durations).tofile(
            f"{self.save_dir}/DQN-test-durations-{self.runid}.csv", sep=',')
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

        state_b, reward_b, new_state_b, done_b = zip(*batch)
        state_b = torch.stack(state_b).to(self.device)
        reward_b = torch.from_numpy(
            np.array(reward_b, dtype=np.float32)[:, None]).to(self.device)
        new_state_b = torch.stack(new_state_b).to(self.device)

        # Use model to predict state values, train prediction against target network
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

    def get_action_and_new_state(self):
        """
        Get potential subsequent states, determine the state with the highest value 
        using the current model, and select the state and requisite action with
        an epsilon greedy strategy. Uses the environment method to generate
        subsequent stats. 

        :return: Tuple of action chosen and new state. 
        """
        action_states = self.env.get_next_states()

        new_actions, new_states = zip(*action_states.items())
        new_states = torch.stack(new_states).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(new_states)[:, 0]
        self.model.train()

        if random.random() <= self.epsilon:
            idx = random.randint(0, len(new_actions) - 1)
        else:
            idx = torch.argmax(predictions).item()

        new_state = new_states[idx, :]
        return new_actions[idx], new_state

    def load(self):
        """
        Load trained or partially trained model from load file specified
        in agent parameters. 
        """
        if self.load_file == None:
            raise ValueError("No load file given")

        if self.load_file[-3:] != ".pt":
            self.model = torch.load(self.load_file).to(self.device)
        else:
            self.model.load_state_dict(
                torch.load(self.load_file))

        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

    def _log(self, ep_score: int):
        """
        Log information about the current epoch to output and TensorBoard.

        :param ep_score: score from the previous episode.
        """
        self.epochs.append(ep_score)
        print(f"Epoch: {self.epoch}, score: {ep_score}")
        self.writer.add_scalar(f'MBDQN-{self.runid}/Lines cleared per epoch',
                               ep_score, self.epoch - 1)
        self.writer.add_scalar(f'MBDQN-{self.runid}/Lines cleared over last 100 timesteps',
                               sum(self.timesteps[-100:]), self.timestep - 1)
        self.writer.add_scalar(
            f'MBDQN-{self.runid}/Epsilon vlaue', self.epsilon, self.epoch - 1)

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
        self.render = args.render
        self.target_network_update = args.target_network_update
        self.saving_interval = args.saving_interval
        self.load_file = args.load_file
        self.one_piece = not args.one_piece

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
        agent = MBDQN(args)
        agent.test()
    else:
        agent = MBDQN(args)
        agent.train()
