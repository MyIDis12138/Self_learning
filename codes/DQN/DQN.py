from operator import ne
from turtle import forward
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from utils import MLP


class Qnet(nn.Module):
    def __init__(
        self, 
        act_dim: int, 
        obs_dim: int, 
        h_dim: int, 
        activation: Optional[nn.Module] = nn.ReLU(),
        layer_num: Optional[int] = 1
        ):
        super().__init__()

        self.layers = MLP(
            input_dim= act_dim + obs_dim,
            output_dim=1,
            hidden_dim= h_dim,
            layer_num=layer_num,
            acti_fn=activation
        )
        self.layers.apply(_init_weights)

    def forward(self, obs: Tensor, action: Tensor):
        assert obs.size(0) == action.size(0)
        net_in = torch.cat(obs, action, dim=1)
        x = self.layers(net_in)
        return x

@torch.no_grad()
def _init_weights(layer: nn.Module):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)


class DQN(object):
    def __init__(self) -> None:
        pass
     
    def select_action(self, state):
        pass

test_obs = torch.randn(3,3,3)
test_act = torch.randn(3,3,3)
net = Qnet(obs_dim=4,act_dim=2,h_dim=5,layer_num=5)
#x = net(test_obs, test_act)
out = torch.cat([test_obs,test_act],dim=1)
print(out.shape)