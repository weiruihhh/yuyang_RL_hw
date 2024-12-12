import argparse

# import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--num-stacks',
        type=int,
        default=4)
    parser.add_argument(
        '--num-steps',
        type=int,
        default=100)
    parser.add_argument(
        '--test-steps',
        type=int,
        default=2000)
    parser.add_argument(
        '--num-frames',
        type=int,
        default=100000)

    ## other parameter
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-img',
        type=bool,
        default=True)
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one eval per n updates (default: None)')
    
    ##new add
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help='折扣率')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='学习率')

    parser.add_argument(
        '--decay_epsilon',
        type=float,
        default=0.1,
        help='epsilon衰减率')
    args = parser.parse_args()

    return args