from lib2to3.pgen2.tokenize import TokenError
import gym
import torch
import numpy as np

env = gym.make('CartPole-v1', new_step_api=True)
s = env.reset(seed=1)   
a = torch.ones(1,1,dtype=torch.int32)
print(a)
action = a.item()
s = env.step(action)