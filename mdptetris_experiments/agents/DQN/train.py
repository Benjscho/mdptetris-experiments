import argparse
import os
import random
import time
from collections import deque 

import gym
import gym_mdptetris
import numpy as np
import torch
from gym_mdptetris.envs import board, piece, tetris
from mdptetris_experiments.agents.DQN.DQ_network import DQ_network
from mdptetris_experiments.agents.linear_agent import LinearGame
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_height", type=int, default=20)
    parser.add_argument("--board_width", type=int, default=10)
    parser.add_argument("--replay_buffer_length", type=int, default=20000)
    parser.add_argument("--reward_buffer_length", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--target_network_update", type=int, default=5)
    parser.add_argument("--epsilon_decay_period", type=int, default=2000)
    parser.add_argument("--state_representation", type=str, default="board-2D")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--comment", type=str, default=None,
                        help="Run comment for TensorBoard writer.")

    args = parser.parse_args()
    return args


def train(args: argparse.Namespace):
    # Set up environment
    env = LinearGame(board_height=args.board_height,
                     board_width=args.board_width)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Writer for TensorBoard
    writer = SummaryWriter(args.log_dir, comment=args.comment)

    # Set up model, network, optimizer, and memory buffer
    model = DQ_network().to(device)
    target = DQ_network().to(device)
    target.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)
    replay_buffer = deque(maxlen=args.replay_buffer_length)
    loss_criterion = nn.MSELoss()
    epochs = np.array([])
    timesteps = np.array([])

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

    state = env.reset()
    if torch.cuda.is_available():
        state = state.cuda()

    epoch = 0
    timestep = 0
    while epoch < args.epochs:
        action_states = env.get_next_states()
        epsilon = args.final_epsilon + (max(args.epsilon_decay_period - epoch, 0) * (
            args.init_epsilon - args.final_epsilon) / args.epsilon_decay_period)
        new_actions, new_states = zip(*action_states.items())
        new_states = torch.stack(new_states)

        if torch.cuda.is_available():
            new_states = new_states.cuda()

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

        new_state = new_states[idx, :]
        if torch.cuda.is_available():
            new_state = new_state.cuda()
        action = new_actions[idx]
        reward, done = env.step(action)
        timestep += 1

        # Append step to buffer
        replay_buffer.append([state, reward, new_state, done])
        timesteps = np.append(timesteps, reward)
        if done:
            episode_score = env.lines_cleared
            state = env.reset()
        else:
            state = new_state
            continue

        if len(replay_buffer) < args.batch_size:
            continue
        epoch += 1
        batch = random.sample(replay_buffer, min(
            len(replay_buffer), args.batch_size))
        state_b, reward_b, new_state_b, done_b = zip(*batch)
        state_b = torch.stack(tuple(state for state in state_b))
        reward_b = torch.from_numpy(
            np.array(reward_b, dtype=np.float32)[:, None])
        new_state_b = torch.stack(tuple(state for state in new_state_b))

        q_vals = target(state_b)
        if torch.cuda.is_available():
            q_vals = q_vals.cuda()
        model.eval()
        with torch.no_grad():
            next_predictions = model(new_state_b)
        model.train()

        y_b = torch.cat(
            tuple(reward if done else reward + args.gamma * prediction for reward, done, prediction in
                  zip(reward_b, done_b, next_predictions))
        )[:, None]

        # Calculate loss and train network
        optimizer.zero_grad()
        loss = loss_criterion(q_vals, y_b)
        loss.backward()
        optimizer.step()

        # Update the target network
        if epoch % args.target_network_update == 0:
            target.load_state_dict(model.state_dict())

        print(f"Epoch: {epoch}, score: {episode_score}")
        writer.add_scalar('Train/Lines cleared per epoch',
                          episode_score, epoch - 1)
        writer.add_scalar('Train/Lines cleared over last 100 timesteps',
                          sum(timesteps[-100:]), timestep - 1)


if __name__ == '__main__':
    # Train
    args = get_args()
    train(args)
