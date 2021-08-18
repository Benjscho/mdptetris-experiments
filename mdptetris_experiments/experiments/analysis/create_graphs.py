import csv
import sys
from argparse import ArgumentParser, Namespace
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from run_arg_parser import get_parser

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


def analyse_MBDQN(run_dirs: List[str]):
    run_args = get_run_args(run_dirs)
    run_timesteps = {}
    run_epochs = {}

    len_t = 0
    len_e = 0

    for dir in run_dirs:
        run_timesteps[dir] = read_result_csv(dir + "/timesteps.csv")
        run_epochs[dir] = read_result_csv(dir + "/epochs.csv")
        len_t = max(len_t, len(run_timesteps[dir]))
        len_e = max(len_e, len(run_epochs[dir]))

    for dir in run_dirs:
        df = pd.DataFrame(run_epochs[dir])
        df = df.astype(float)
        # Grouping results: https://stackoverflow.com/a/36810658/14354978
        df = smooth_data(df)
        plt.plot(df)
    plt.show()


def smooth_data(df: pd.DataFrame, alpha: float = 0.9, grouping: int = 100):
    tdf = df.groupby(np.arange(len(df))//100).mean()
    return tdf.ewm(alpha=(1 - 0.9)).mean()


def analyse_PPO(run_dirs):
    pass


def main():

    analyse_MBDQN(MBDQN_RUNS)


if __name__ == '__main__':
    run_dirs = sys.argv[1:]
    main()
