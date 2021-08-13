import csv
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from run_arg_parser import get_parser


# graph reward against timesteps
def reward_over_time(reward):
    timesteps = [i for i in range(len(reward))]
    plt.plot(timesteps, reward)
    plt.show()


def get_run_args(run_dirs):
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

def analyse_MBDQN(run_dirs): 
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
        print()

        # Grouping results: https://stackoverflow.com/a/36810658/14354978 
        df = df.groupby(np.arange(len(df))//100).mean()
        print(df)
        plt.plot(df.ewm(alpha=(1 - 0.9)).mean())
    plt.show()

def smooth_data(df: pd.DataFrame, alpha: float=0.9, grouping: int=100):
    tdf = df.groupby(np.arange(len(df))//100).mean()
    return tdf.ewm(alpha=(1 - 0.9)).mean()

def analyse_PPO(run_dirs): 
    pass

def main(run_dirs):

    analyse_MBDQN(run_dirs)
    pass

if __name__=='__main__':
    run_dirs = sys.argv[1:]
    main(run_dirs)
