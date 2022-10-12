
import torch 
import torch.nn as nn
from torch import Tensor
from torchsummary import summary

def MLP(input_dim, output_dim, hidden_dim, layer_num, acti_fn: nn.Module):
    print(f"in_dim: {input_dim}, out_dim: {output_dim}, hidden_dim: {hidden_dim}, layer_num: {layer_num}")
    assert layer_num>=0, layer_num
    blocks = []
    channels = [input_dim] + [hidden_dim] * (layer_num - 1) + [output_dim]
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
        blocks.append(nn.Linear(in_channels,out_channels))
        if acti_fn  is not None:
            blocks.append(acti_fn)
    return sequ_net(blocks)


def sequ_net(layers: list) -> nn.Sequential:
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    return seq

