import sys
from typing import runtime_checkable
from matplotlib import pyplot as plt
from argparse import ArgumentParser, Namespace
from run_arg_parser import get_parser

# graph reward against timesteps
def reward_over_time(reward, ):
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

def analyse_MBDQN(run_dirs): 
    run_args = get_run_args(run_dirs)
    for dir in run_dirs: 
        runinfo = dir 
        timesteps = runinfo + "timesteps.csv"
        epochs = runinfo + "epochs.csv"
    pass

def analyse_PPO(run_dirs): 
    pass

def main(run_dirs):
    pass

if __name__=='__main__':
    run_dirs = sys.argv[1:]
    main(run_dirs)