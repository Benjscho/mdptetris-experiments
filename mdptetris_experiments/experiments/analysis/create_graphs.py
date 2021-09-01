import ast
import csv
import sys
from argparse import ArgumentParser, Namespace
from typing import List

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from run_arg_parser import get_parser

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


SAVE_DIR = "/Users/crow/Desktop/bath-msc-main/cm50170-Diss/Dissertation/resources/"
ROOT_RUN_DIR = "/Users/crow/Desktop/bath-msc-main/cm50170-Diss/experiment-results/main-runs/run-info-"
MBDQN_RUNS = [ROOT_RUN_DIR +
              i for i in ["20210809T161605Z", "20210811T071836Z"]]
MBDQN_1PIECE_RUNS = [ROOT_RUN_DIR +
                     i for i in ["20210713T101844Z", "20210803T070646Z"]]
PPO_RUNS = [ROOT_RUN_DIR + i for i in ["20210730T093417Z", "20210802T205340Z"]]
MBDQN_RUNS_SUPPLEMENTARY = [ROOT_RUN_DIR +
              i for i in ["20210809T092959Z", "20210809T095013Z"]]
MBDQN_RUNS_SUPPLEMENTARY_2 = [ROOT_RUN_DIR +
              i for i in ["20210809T112302Z"]]
PPO_CLIP_RUNS = [ROOT_RUN_DIR +
              i for i in ["20210828T071816Z", "20210826T082657Z", "20210827T100614Z"]]
            


def get_run_args(run_dirs: List[str]) -> dict:
    """
    Method to parse the run arguments saved in MBDQN run directories. 

    :param run_dirs: List of directories to parse run args for

    :return: Returns a dictionary of run arguments 
    """
    parser = get_parser()
    run_args = {}
    for dir in run_dirs:
        with open(dir + "/args.txt") as f:
            ns = f.read()
            args, unknown = parser.parse_known_args(namespace=eval(ns))
        run_args[dir] = args
    return run_args


def read_result_csv(file_path: str) -> List:
    """
    Method to read in the first line of a csv to an array. 
    
    :param file_path: Path to csv file. 

    :return: Returns array containing the first line of the csv
    """
    with open(file_path, mode='r') as file:
        result = csv.reader(file)
        t = [l for l in result]
    return t[0]


def analyse_MBDQN(run_dirs: List[str], title: str, save_file: str, alpha: float=0.8, grouping: int=20):
    """
    Method to take in a list of MBDQN run directories, iterate through them
    to collect run information, and plot the resulting data. 

    :param run_dirs: list of run result directories
    :param title: Title for the resulting graph
    :param save_file: Name for the save file
    """
    run_args = get_run_args(run_dirs)
    run_timesteps = {}
    run_epochs = {}

    for dir in run_dirs:
        run_timesteps[dir] = read_result_csv(dir + "/timesteps.csv")
        run_epochs[dir] = read_result_csv(dir + "/epochs.csv")

    for dir in run_dirs:
        print()
        print(f"{save_file} Hyperparameters:")
        for key in run_args[dir].__dict__:
            if run_args[dir].__dict__[key]:
                print(f"{key} & {run_args[dir].__dict__[key]} \\\\".capitalize().replace(
                    "_", " "))

    plt.figure(save_file)
    plt.title(title)
    for dir in run_dirs:
        df = pd.DataFrame(run_epochs[dir])
        df = df.astype(float)
        df = smooth_data(df, alpha, grouping)
        plt.plot([i*grouping for i in range(len(df))], df,
                 label=run_args[dir].state_rep.capitalize())
    plt.legend(title="State representation")
    plt.xlabel("Training epoch")
    plt.ylabel(f"Average reward per episode, grouped over {grouping} epochs")
    plt.savefig(SAVE_DIR + save_file + ".png")


def smooth_data(df: pd.DataFrame, alpha: float = 0.9, grouping: int = 100):
    """
    Method to group results of a dataframe and smooth with an alpha factor. 
    Attribution: https://stackoverflow.com/a/36810658/14354978
    """
    tdf = df.groupby(np.arange(len(df))//grouping).mean()
    return tdf.ewm(alpha=(1 - alpha)).mean()


def analyse_PPO(run_dirs: List[str], title: str, save_file: str):
    """
    Method to take in a list of PPO run directories, iterate through them
    to collect run information, and plot the resulting data. 

    :param run_dirs: list of run result directories
    :param title: Title for the resulting graph
    :param save_file: Name for the save file
    """
    run_args = {}
    for dir in run_dirs:
        with open(dir + "/args.txt") as f:
            data = f.read()
            run_args[dir] = ast.literal_eval(data)

    avg_rewards = {}
    timesteps = {}

    for dir in run_dirs:
        timesteps[dir] = pd.read_csv(dir + "/rewards.csv")

    # Print hyperparams
    for dir in run_dirs:
        print()
        print(f"{save_file} Hyperparameters:")
        for key in run_args[dir]:
            if run_args[dir][key]:
                print(
                    f"{key} & {run_args[dir][key]} \\\\".capitalize().replace("_", " "))

    plt.figure(save_file)
    plt.title(title)
    grouping = 1
    for dir in run_dirs:
        df = timesteps[dir]
        df = smooth_data(df, 0.8, grouping)
        plt.plot(df['Step'], df['Value'], label=run_args[dir]['board_height'])
    plt.legend(title="Board height")
    plt.xlabel("Time steps")
    plt.ylabel(f"Average reward per time step")
    plt.savefig(SAVE_DIR + save_file + ".png")


def analyse_PPO_clip(run_dirs: List[str], title: str, save_file: str, limit: int = 50_000_000):
    """
    Method to take in a list of PPO run directories, iterate through them
    to collect run information, and plot the resulting data. 

    :param run_dirs: list of run result directories
    :param title: Title for the resulting graph
    :param save_file: Name for the save file
    """
    run_args = {}
    for dir in run_dirs:
        with open(dir + "/args.txt") as f:
            data = f.read()
            run_args[dir] = ast.literal_eval(data)

    avg_rewards = {}
    timesteps = {}

    for dir in run_dirs:
        timesteps[dir] = pd.read_csv(dir + "/rewards.csv")

    # Print hyperparams
    for dir in run_dirs:
        print()
        print(f"{save_file} Hyperparameters:")
        for key in run_args[dir]:
            if run_args[dir][key]:
                print(
                    f"{key} & {run_args[dir][key]} \\\\".capitalize().replace("_", " "))

    plt.figure(save_file)
    plt.title(title)
    grouping = 1
    for dir in run_dirs:
        df = timesteps[dir]
        df = smooth_data(df, 0.8, grouping)
        plt.plot(df['Step'], df['Value'], label=run_args[dir]['clip'])
    plt.legend(title="Clip value")
    plt.xlabel("Time steps")
    plt.xlim([0,limit])
    plt.ylabel(f"Average reward per time step")
    plt.savefig(SAVE_DIR + save_file + ".png")


def main():
    """
    Define main method to analyse PPO, and MBDQN runs specified in the global
    vars above. 
    """
    analyse_PPO(PPO_RUNS, "PPO Learning rate", "ppo")
    analyse_MBDQN(MBDQN_1PIECE_RUNS,
                  "(a) MBDQN learning rate for single piece episodes", "mbdqn-1piece")
    analyse_MBDQN(
        MBDQN_RUNS, "(b) MBDQN learning rate on standard Tetris", "mbdqn")
    analyse_MBDQN([MBDQN_RUNS[0]], "MBDQN learning rate 1D only", "mbdqn-1d", 0.8, 20)
    analyse_MBDQN(MBDQN_RUNS_SUPPLEMENTARY, "MBDQN learning rate, longer duration (a)", "mbdqn-10_000", 0.8, 20)
    analyse_MBDQN(MBDQN_RUNS_SUPPLEMENTARY_2, "MBDQN learning rate, longer duration (b)", "mbdqn-100_000", 0.8, 20)
    analyse_PPO_clip(PPO_CLIP_RUNS, "PPO learning rate as clip value varies", "ppo-clip")


if __name__ == '__main__':
    main()
