
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils import MLP
from memory import DQNBuffer
import gym

class Qnet(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int, 
        h_dim: int, 
        activation: Optional[nn.Module] = nn.ReLU(),
        layer_num: Optional[int] = 3
        ):
        super().__init__()

        self.layers = MLP(
            input_dim= input_dim,
            output_dim=output_dim,
            hidden_dim= h_dim,
            layer_num=layer_num,
            acti_fn=activation
        )
        self.layers.apply(_init_weights)

    def forward(self, obs: Tensor):
        return self.layers(obs)

@torch.no_grad()
def _init_weights(layer: nn.Module):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)


class DQN(object):
    def __init__(
        self,
        env: gym.Env,
        batch_size: int, 
        gamma: float, 
        learning_rate: float, 
        buffer_sizes:int,
        update_frequency: int,
        q_net_h_dim:int,
        q_net_layer_num:int
    ):
        self.env = env
        self.buffer_sizes = buffer_sizes
        
        self.q_net_h_dim = q_net_h_dim
        self.q_net_layer_num = q_net_layer_num
        self.batch_size = batch_size
        self._build_memory()
        self._build_q_net()
        
        self.gamma = gamma
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.update_frequency = update_frequency
        self.n_updates = 0

    def _build_memory(self):
        self.memory = DQNBuffer(
            buffer_size=self.buffer_sizes,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
    
    def _build_q_net(self):
        self.q_net = Qnet(
            input_dim=self.env.observation_space.shape[0],
            output_dim=self.env.action_space.n,
            h_dim=self.q_net_h_dim,
            layer_num=self.q_net_layer_num
        )
        self.target_q_net = deepcopy(self.q_net)

    @torch.no_grad()
    def select_action(self, obs):
        q_values = self.q_net(torch.as_tensor(obs))
        action = q_values.max(dim=0)[1].item()
        return action

    def update(self):
        try:
            bs, ba, br, bs_, bd = self.memory.sample(self.batch_size)
        except ValueError:
            # Continue when samples less than a batch
            return

        with torch.no_grad():
            qs = self.target_q_net(bs_)
            max_qs,_ = qs.max(dim=2)
            target_q = br + (1-bd)*self.gamma*max_qs
        
        current_q = self.q_net(bs)
        Q_values = torch.gather(input=current_q, dim=2, index=ba)
        
        loss = F.smooth_l1_loss(target_q, Q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.n_updates += 1

        if self.n_updates%self.update_frequency == 0:
            self.target_q_net = deepcopy(self.q_net)

