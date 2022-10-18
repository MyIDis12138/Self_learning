import sys,os
from datetime import datetime
from tkinter import E
from typing import Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
curr_time = datetime.now().strftime("%Y%m%d-%H%M%S") 

import DDPG
from DDPG.ddpg import DDPG
from DDPG.nn import Actor, Critic
from DDPG.memory import PER, UER
from DDPG.noise import Gaussian
from DQN.utils import get_action_dim, get_obs_shape

import gym
import torch
from torch.nn import functional as F
import numpy as np

cfg = {
    "env_name": 'MountainCarContinuous-v0',
    "episodes": 200,
    "n_steps": 2000,
}
agent_cfg={
    "memory_sizes": 100000,
    "dicount_factor":0.99,
    "polyak": 0.99,
    "batch_sizes":128
}

def _build_agent(env:gym.Env, agent_cfg:Dict):
    actor = Actor(
        observation_dim=env.observation_space.shape[0],
        action_dim=get_action_dim(env.action_space),
        hidden_dims=[128,128]
    )
    critic = Critic(
        observation_dim=env.observation_space.shape[0],
        action_dim=get_action_dim(env.action_space),
        hidden_dims=[128,128]
    )
    agent = DDPG(
        actor= actor,
        critic=critic,
        actor_optimiser=torch.optim.Adam,
        critic_optimiser=torch.optim.Adam,
        memory=PER(capacity=agent_cfg["memory_sizes"], alpha=0.5),
        discount_factor=agent_cfg["dicount_factor"],
        batch_size=agent_cfg["batch_sizes"],
        polyak=agent_cfg["polyak"],
        noise=Gaussian(0.01)
    )
    return agent

def into_tensor(s, a, r, s_,done):
    s = torch.as_tensor(s)
    a = torch.as_tensor(a)
    r = torch.as_tensor(r)
    s_ = torch.as_tensor(s_)
    done = torch.as_tensor(done)
    return s,a,r,s_,done


def main():
    env = gym.make(cfg["env_name"])
    print(cfg["env_name"])
    agent:DDPG = _build_agent(env, agent_cfg)
    print("agent built")

    s,_ = env.reset()
    reward_buffer = []
    for episode_i in range(cfg["episodes"]):
        episode_reward = 0    
        for step_i in range(cfg["n_steps"]):
            a = agent.compute_action(torch.as_tensor(s))
            s_, r, done, _, info = env.step(np.array(a))
            s, a, r, s_, done = into_tensor(s,a,r,s_,done)
            agent.memory.push(action=a, next_state=s_, state=s, done=done,reward=r)
            
            episode_reward += r
            agent.update()
            
            s = s_
            if done:
                s,_ = env.reset()
                break
        
        if(len(reward_buffer)>0 and episode_reward>max(reward_buffer)):
            print(f"model need update")
            
        print(f"epsisode:{episode_i}, reward:{episode_reward}")
        episode_reward = 0

if __name__=="__main__":
    main()