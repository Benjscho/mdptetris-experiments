
import argparse

import argparse

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default=None)

    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = get_args()
