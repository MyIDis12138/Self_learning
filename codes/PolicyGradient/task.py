import imp
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path



import gym 
import torch
import datetime
import argparse
from itertools import count

from pg import PolicyGradient
from common.utils import save_results, make_dir
from common.utils import plot_rewards

class Config:
    def __init__(self) -> None:
        """
        define hyperparameters
        """
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
        parser = argparse.ArgumentParser(description="hyperparameters")      
        parser.add_argument('--algo_name',default='PolicyGradient',type=str,help="name of algorithm")
        parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
        parser.add_argument('--train_eps',default=300,type=int,help="episodes of training")
        parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
        parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
        parser.add_argument('--lr',default=0.01,type=float,help="learning rate")
        parser.add_argument('--batch_size',default=8,type=int)
        parser.add_argument('--hidden_dim',default=36,type=int)
        parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
        parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                '/' + curr_time + '/results/' )
        parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                '/' + curr_time + '/models/' ) # path to save models
        parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")           
        args = parser.parse_args()   
        self.args = args

def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env_name)
    env.seed(seed)
    n_states = env.observation_space.shape[0]
    agent = PolicyGradient(n_states, cfg)
    return env, agent


