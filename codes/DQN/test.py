from collections import namedtuple
import gym
import torch
import numpy as np
from DDQN import Qnet

net = Qnet(
    input_dim=4,
    output_dim=2,
    hidden_layers=[32,32]
)

x = net(torch.randn(5,4))
x = x.max(dim=1)
print(x)