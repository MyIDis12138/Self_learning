from typing import Union, Tuple, Dict
import torch.nn as nn
from gym import spaces
import numpy as np


def MLP(input_dim, output_dim, hidden_layers:list, acti_fn: nn.Module, activate_last: bool = False):
    print(f"in_dim: {input_dim}, out_dim: {output_dim}, layer_num: {len(hidden_layers)}")
    blocks = []
    channels = [input_dim] + hidden_layers + [output_dim]
    for in_channels, out_channels in zip(channels[:-1], channels[1:]):
        blocks.append(nn.Linear(in_channels,out_channels))
        if acti_fn  is not None:
            blocks.append(acti_fn)
    if (~activate_last):
        blocks = blocks[:-1]
    return sequential_net(blocks)


def sequential_net(layers: list) -> nn.Sequential:
    assert isinstance(layers, list)
    return nn.Sequential(*layers)

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return len(action_space.nvec)
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).
    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (len(observation_space.nvec), )
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")