import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
from gym_mdptetris.envs import board, piece, tetris
from mdptetris_experiments.agents.FFNN import NN1D, NNHeuristic
from mdptetris_experiments.agents.linear_agent import LinearGame, LinearGameStandard
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
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

def save(save_dir, model, epochs, timesteps):
    torch.save(model, f"{save_dir}/model")
    epochs.tofile(f"{save_dir}/epochs.csv", sep=',')
    timesteps.tofile(f"{save_dir}/timesteps.csv", sep=',')

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

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

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
    epochs = np.array([], dtype=np.int32)
    timesteps = np.array([], dtype=np.int32)

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
    while epoch < args.epochs:
        action_states = env.get_next_states()
        epsilon = args.final_epsilon + (max(args.epsilon_decay_period - epoch, 0) * (
            args.init_epsilon - args.final_epsilon) / args.epsilon_decay_period)
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
        timestep += 1

        # Append step to buffer and save reward
        # If I want to save more infrequent reward I can use the buffer to batch
        # results across timesteps.
        replay_buffer.append([state, reward, new_state, done])
        timesteps = np.append(timesteps, reward)

        # Skip epoch increment if episode is not done. If done, record episode
        # score and reset env. 
        if done:
            episode_score = env.lines_cleared
            state = env.reset().to(device)
        else:
            state = new_state
            continue

        # Skip training until memory buffer exceeds min timesteps
        if len(replay_buffer) < args.training_start:
            continue

        epoch += 1

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

        y_b = torch.cat(
            tuple(reward if done else reward + args.gamma * prediction for reward, done, prediction in
                  zip(reward_b, done_b, next_predictions))
        )[:, None].to(device)

        # Calculate loss and train network
        optimizer.zero_grad()
        loss = loss_criterion(q_vals, y_b)
        loss.backward()
        optimizer.step()

        # Update the target network
        if epoch % args.target_network_update == 0:
            target.load_state_dict(model.state_dict())

        epochs = np.append(epochs, episode_score)
        print(f"Epoch: {epoch}, score: {episode_score}")
        writer.add_scalar(f'Train-{runid}/Lines cleared per epoch',
                          episode_score, epoch - 1)
        writer.add_scalar(f'Train-{runid}/Lines cleared over last 100 timesteps',
                          sum(timesteps[-100:]), timestep - 1)

        # On interval, save model and current results to csv 
        if epoch % args.saving_interval == 0:
            save(save_dir, model, epochs, timesteps)

    # Save on completion
    save(save_dir, model, epochs, timesteps)


if __name__ == '__main__':
    # Train the model
    args = get_args()
    train(args)
