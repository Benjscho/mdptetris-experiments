import argparse
import random
import time

import torch
from mdptetris_experiments.agents.FFNN import NN1D, NNHeuristic
from mdptetris_experiments.agents.linear_agent import (LinearGame,
                                                       LinearGameStandard)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_height", type=int, default=20)
    parser.add_argument("--board_width", type=int, default=10)
    parser.add_argument("--state_rep", type=str, default="heuristic")
    parser.add_argument("--load_file", type=str, default=None,
                        help="Path to trained model")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=10000)
    parser.add_argument("--save_dir", type=str,
                        default=f"runs/run-info")

    args = parser.parse_args()
    return args

state_rep = {
    "heuristic": [NNHeuristic, LinearGame],
    "1D": [NN1D, LinearGameStandard]
}

def test(args: argparse.Namespace):
    env = state_rep[args.state_rep][1](board_height=args.board_height,
                                       board_width=args.board_width)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    runid = time.strftime('%Y%m%dT%H%M%SZ')
    save_dir = f"{args.save_dir}-{runid}"
    model = torch.load(args.load_file)
    # Put model in eval mode (no training during test)
    model.eval()

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

    episode = 0 
    timestep = 0 
    ep_score = 0

    while timestep < args.total_timesteps:
        action_states = env.get_next_states()
        new_actions, new_states = zip(*action_states.items())
        new_states = torch.stack(new_states).to(device)

        with torch.no_grad():
            predictions = model(new_states)[:, 0]

        idx = torch.argmax(predictions).item()

        new_state = new_states[idx, :].to(device)
        action = new_actions[idx]
        reward, done = env.step(action)
        ep_score += reward
        timestep += 1

        if done: 
            tot_ep_score = ep_score
            ep_score = 0 
            state = env.reset().to(device)
        else:
            state = new_state
            continue