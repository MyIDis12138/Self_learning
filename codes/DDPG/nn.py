from functools import reduce
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Actor(nn.Module):

    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: Iterable[int]):
        super(Actor, self).__init__()

        dimensions = [observation_dim] + list(hidden_dims) + [action_dim]

        self.layers = nn.ModuleList( 
            [ nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dimensions, dimensions[1:]) ])

        self.layers.apply(_init_weights)

    def forward(self, state: Tensor):

        # TODO: Use nn.ModuleDict for selecting diff activation func

        # FIXME: torch.jit.script raises torch.jit.frontend.UnsupportedNodeError: Lambda aren't supported
        action = torch.tanh(self.layers[-1](
            reduce( lambda activation, layer: F.relu(layer(activation)), self.layers[:-1], state )))
            
        return action


class Critic(nn.Module):

    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: Iterable[int]):
        super(Critic, self).__init__()

        dimensions = [observation_dim + action_dim] + list(hidden_dims) + [1]

        self.layers = nn.ModuleList(
            [ nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dimensions, dimensions[1:]) ])

        self.layers.apply(_init_weights)

    def forward(self, state: Tensor, action: Tensor):

        value = self.layers[-1](
            reduce( lambda activation, layer: F.relu(layer(activation)), self.layers[:-1], torch.cat([state, action], dim=1) ))

        return value


@torch.no_grad()
def _init_weights(layer: nn.Module):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)

