import argparse
from matplotlib import pyplot as plt

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default=None)

    args = parser.parse_args()
    return args


# graph reward against timesteps
def reward_over_time(reward):
    timesteps = [i for i in range(len(reward))]
    plt.plot(timesteps, reward)
    plt.show()

# graph epoch against reward

if __name__=='__main__':
    args = get_args()
