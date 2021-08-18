import csv
import sys
from argparse import ArgumentParser, Namespace
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

from run_arg_parser import get_parser

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


ROOT_RUN_DIR = "/Users/crow/Desktop/bath-msc-main/cm50170-Diss/experiment-results/main-runs/run-info-"
MBDQN_RUNS = [ROOT_RUN_DIR + i for i in ["20210809T161605Z", "20210811T071836Z"]]
MBDQN_1PIECE_RUNS = [ROOT_RUN_DIR + i for i in ["20210713T101844Z", "20210803T070646Z"]]
PPO_RUNS = [ROOT_RUN_DIR + i for i in ["20210730T093417Z", "20210802T205340Z"]]


# graph reward against timesteps
def reward_over_time(reward):
    timesteps = [i for i in range(len(reward))]
    plt.plot(timesteps, reward)
    plt.show()


def get_run_args(run_dirs: List[str]):
    parser = get_parser()
    run_args = {}
    for dir in run_dirs:
        with open(dir + "/args.txt") as f:
            ns = f.read()
            args, unknown = parser.parse_known_args(namespace=eval(ns))
        run_args[dir] = args
    return run_args


def read_result_csv(file_path):
    with open(file_path, mode='r') as file:
        result = csv.reader(file)
        t = [l for l in result]
    return t[0]


def analyse_MBDQN(run_dirs: List[str], title: str):
    run_args = get_run_args(run_dirs)
    run_timesteps = {}
    run_epochs = {}

    for dir in run_dirs:
        run_timesteps[dir] = read_result_csv(dir + "/timesteps.csv")
        run_epochs[dir] = read_result_csv(dir + "/epochs.csv")

    plt.title(title)
    grouping = 20
    for dir in run_dirs:
        df = pd.DataFrame(run_epochs[dir])
        df = df.astype(float)
        df = smooth_data(df, 0.8, grouping)
        plt.plot([i*grouping for i in range(len(df))], df, label=run_args[dir].state_rep.capitalize())
    plt.legend()
    plt.xlabel("Training epoch")
    plt.ylabel(f"Average reward per {grouping} epochs")
    plt.show()


def smooth_data(df: pd.DataFrame, alpha: float = 0.9, grouping: int = 100):
    """
    Method to group results of a dataframe and smooth with an alpha factor. 
    Attribution: https://stackoverflow.com/a/36810658/14354978
    """
    tdf = df.groupby(np.arange(len(df))//grouping).mean()
    return tdf.ewm(alpha=(1 - alpha)).mean()


def analyse_PPO(run_dirs: List[str], title: str):
    run_args = get_run_args(run_dirs)
    avg_rewards = {}
    timesteps = {}

    for dir in run_dirs:
        timesteps[dir] = read_result_csv(dir + "/epoch_timesteps.csv")
        avg_rewards[dir] = read_result_csv(dir + "/avg_ep_rewards.csv")

    plt.title(title)
    grouping = 20
    for dir in run_dirs:
        df = pd.DataFrame([avg_rewards[dir], timesteps[dir]])
        df = df.astype(float)
        print(df)
        df = smooth_data(df, 0.8, grouping)
        print(df)
        plt.plot([i*grouping for i in range(len(df))], df, label=run_args[dir].state_rep.capitalize())
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel(f"Average reward per {grouping} epochs")
    plt.show()


def main():
    analyse_PPO(PPO_RUNS, "PPO Learning rate")
    analyse_MBDQN(MBDQN_1PIECE_RUNS, "(a) MBDQN learning for one piece per episode")
    analyse_MBDQN(MBDQN_RUNS, "(b) MBDQN learning on standard Tetris")


if __name__ == '__main__':
    run_dirs = sys.argv[1:]
    main()
