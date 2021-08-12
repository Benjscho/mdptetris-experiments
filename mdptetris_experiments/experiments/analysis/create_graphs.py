import sys
from matplotlib import pyplot as plt
from argparse import ArgumentParser, Namespace
from run_arg_parser import get_parser

# graph reward against timesteps
def reward_over_time(reward, ):
    timesteps = [i for i in range(len(reward))]
    plt.plot(timesteps, reward)
    plt.show()


def main(run_dirs): 
    parser = get_parser()
    for dir in run_dirs: 
        runinfo = dir 
        timesteps = runinfo + "timesteps.csv"
        epochs = runinfo + "epochs.csv"
        with open(dir + "/args.txt") as f:
            ns = f.read()
            args, unknown = parser.parse_known_args(namespace=eval(ns))
        print(args)
        print(unknown)
    pass


if __name__=='__main__':
    run_dirs = sys.argv[1:]
    print(run_dirs)
    main(run_dirs)