import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
from gym_mdptetris.envs import board, piece, tetris
from mdptetris_experiments.agents.FFNN import NN1D, NNHeuristic
from mdptetris_experiments.agents.linear_agent import (LinearGame,
                                                       LinearGameStandard)
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--test", action=argparse.BooleanOptionalAction)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction)
    parser.add_argument("--board_height", type=int, default=20)
    parser.add_argument("--board_width", type=int, default=10)
    parser.add_argument("--replay_buffer_length", type=int, default=20000)
    parser.add_argument("--training_start", type=int, default=2000,
                        help="Minimum timesteps for training to start.")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="Optimiser learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
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


state_rep = {
    "heuristic": [NNHeuristic, LinearGame],
    "1D": [NN1D, LinearGameStandard]
}


class TD_Lambda:
    def __init__(self, args: argparse.Namespace):
        """
        Class that implements a model-based DQN agent to learn a game of Tetris.
        The model for the environment is provided in the linear_agent file, 
        which allows generation of subsequent states, and retrieval of their
        representation as either the full board, or as a set of features. 

        :param args: A Namespace object containing experiment hyperparameters
        """
        self.env = state_rep[args.state_rep][1](board_height=args.board_height,
                                                board_width=args.board_width)

        self._init_hyperparams(args)

        # Initialise models
        input_dims = 6 if args.state_rep == "heuristic" else args.board_height * args.board_width
        if args.load_file != None:
            self.model = torch.load(args.load_file).to(self.device)
        else:
            self.model = state_rep[args.state_rep][0](
                input_dims).to(self.device)
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

            reward, done = self.env.step(action)
            if self.render:
                self.env.render()
            ep_score += reward
            self.timestep += 1

            self.replay_buffer.append([state, reward, new_state, done])
            self.timesteps.append(reward)
            if done:
                self.update_model()
                if self.epoch > 0:
                    self._log(ep_score)
                ep_score = 0
                state = self.env.reset()
            else:
                state = new_state

    def test(self, nb_episodes: int=1000, render: bool=False):
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
                action, _ = self.get_action_and_new_state()
                reward, done = self.env.step(action)
                if render:
                    self.env.render()
                if timesteps % 5000 == 0:
                    print(ep_score, timesteps)
                ep_score += reward
                timesteps += 1

            episode_rewards.append(ep_score)
            episode_durations.append(timesteps)
            print(f"Episode: {i}, Episode reward: {ep_score}, Episode duration: {timesteps}")
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

        state_b, reward_b, new_state_b, done_b = zip(*batch)
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
        self.render = args.render
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


def save(save_dir: str, model: nn.Module, epochs: list, timesteps: list):
    """
    Method to save the current model state, the current saved epoch rewards, and
    the current saved timestep rewards.

    :param save_dir: The directory to save the results in, typically using a runID
    :param model: The current model of the run
    :param epochs: The array of epoch data
    :param timesteps: The array of timestep data
    """
    torch.save(model, f"{save_dir}/model")
    np.array(epochs).tofile(f"{save_dir}/epochs.csv", sep=',')
    np.array(timesteps).tofile(f"{save_dir}/timesteps.csv", sep=',')


def train(args: argparse.Namespace):
    """
    Method that initialises a network with a set of arguments and trains
    it on a given Tetris environment.

    Attribution: This code implements the DQN algorithm in Mnih, V.,
    Kavukcuoglu, K., Silver, D. et al. Human-level control through deep
    reinforcement learning. Nature 518, 529–533 (2015).  
    
    :param args: A namespace containing the hyperparameters and arguments
        for training the model.
    """
    # Set up environment
    env = state_rep[args.state_rep][1](board_height=args.board_height,
                                       board_width=args.board_width)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    runid = time.strftime('%Y%m%dT%H%M%SZ')
    save_dir = f"{args.save_dir}-{runid}"
    # Writer for TensorBoard
    writer = SummaryWriter(args.log_dir, comment=f"{args.comment}-{runid}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(f"{save_dir}/args.txt", 'w') as f:
        f.write(str(args))

    # Set up model, network, optimizer, and memory buffer
    input_dims = 6 if args.state_rep == "heuristic" else args.board_height * args.board_width
    if args.load_file != None:
        model = torch.load(args.load_file)
    else:
        model = state_rep[args.state_rep][0](input_dims).to(device)
    target = state_rep[args.state_rep][0](input_dims).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)
    replay_buffer = deque(maxlen=args.replay_buffer_length)
    loss_criterion = nn.MSELoss()
    epochs = []
    timesteps = []
    epsilon = args.init_epsilon
    epsilon_decay_rate = (args.init_epsilon -
                          args.final_epsilon) / args.epsilon_decay_period

    # Seed randomness
    if args.seed == None:
        seed = int(time.time())
        random.seed(seed)
        env.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
    else:
        random.seed(args.seed)
        env.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        else:
            torch.manual_seed(args.seed)

    state = env.reset().to(device)

    epoch = 0
    timestep = 0
    temp_ep_score = 0
    while epoch < args.epochs:
        action_states = env.get_next_states()
        new_actions, new_states = zip(*action_states.items())
        new_states = torch.stack(new_states).to(device)

        # Predict values of next states
        model.eval()
        with torch.no_grad():
            predictions = model(new_states)[:, 0]
        model.train()

        # Select next state with epsilon greedy strategy
        if random.random() <= epsilon:
            idx = random.randint(0, len(new_actions) - 1)
        else:
            idx = torch.argmax(predictions).item()

        new_state = new_states[idx, :].to(device)
        action = new_actions[idx]
        reward, done = env.step(action)
        temp_ep_score += reward
        timestep += 1

        # Append step to buffer and save reward
        # If I want to save more infrequent reward I can use the buffer to batch
        # results across timesteps.
        replay_buffer.append([state, reward, new_state, done])
        timesteps.append(reward)

        # Skip epoch increment if episode is not done. If done, record episode
        # score and reset env.
        if done:
            episode_score = temp_ep_score
            temp_ep_score = 0
            state = env.reset().to(device)
        else:
            state = new_state
            continue

        # Skip training until memory buffer exceeds min timesteps
        if len(replay_buffer) < args.training_start:
            continue

        # Epoch increase and decrement epsilon
        epoch += 1
        epsilon -= epsilon_decay_rate
        epsilon = max(epsilon, args.final_epsilon)

        batch = random.sample(replay_buffer, min(
            len(replay_buffer), args.batch_size))
        state_b, reward_b, new_state_b, done_b = zip(*batch)
        state_b = torch.stack(state_b).to(device)
        reward_b = torch.from_numpy(
            np.array(reward_b, dtype=np.float32)[:, None]).to(device)
        new_state_b = torch.stack(new_state_b).to(device)

        # Use model to judge state values, train prediction against target network
        q_vals = model(state_b).to(device)
        with torch.no_grad():
            next_predictions = target(new_state_b)

        y_b = []
        for reward, done, prediction in zip(reward_b, done_b, next_predictions):
            y_b.append(reward if done else reward + args.gamma*prediction)
        y_b = torch.cat(y_b).to(device)

        # Calculate loss and train network
        optimizer.zero_grad()
        loss = loss_criterion(q_vals, y_b)
        loss.backward()
        optimizer.step()

        # Update the target network
        if epoch % args.target_network_update == 0:
            target.load_state_dict(model.state_dict())

        epochs.append(episode_score)
        print(f"Epoch: {epoch}, score: {episode_score}")
        writer.add_scalar(f'Train-{runid}/Lines cleared per epoch',
                          episode_score, epoch - 1)
        writer.add_scalar(f'Train-{runid}/Lines cleared over last 100 timesteps',
                          sum(timesteps[-100:]), timestep - 1)
        writer.add_scalar(f'Train-{runid}/Epsilon vlaue', epsilon, epoch - 1)

        # On interval, save model and current results to csv
        if epoch % args.saving_interval == 0:
            save(save_dir, model, epochs, timesteps)

    # Save on completion
    save(save_dir, model, epochs, timesteps)


if __name__ == '__main__':
    # Train the model
    args = get_args()

    if args.test:
        assert args.load_file != None
        agent = TD_Lambda(args)
        agent.test(render=True)
    else:
        agent = TD_Lambda(args)
        agent.train()
