import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='RL_algorithms with gym')
parser.add_argument('--env', type=str, default='Pendulum-v0', help='training Environment')
parser.add_argument('--algo', type=str, required=True, help='The RL algorithm.')
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--load', type=str, default=None, help='mode to load')
parser.add_argument('--seed', type=int, default=0, help='seed for random number ')
parser.add_argument('--episdo', type=int, default=100, help='the episodes to run')
parser.add_argument('--eval_per_train', type=int, default=100, help='evaluation number per training')
parser.add_argument('--max_step', type=int, default=200, help='max episode step')
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu', help='device to train')
args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
 
def import_algo():
    if args.algo == 'ppo':
        from PPO.ppo import Agent

