import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


import gym
from gym.spaces import Discrete, Box
from typing import Optional

from copy import deepcopy
from utils import MLP
from memory import UER, DQNBuffer


class Qnet(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,  
        activation: Optional[nn.Module] = nn.ReLU(),
        hidden_layers:list = [32,32]
        ):
        super().__init__()

        self.layers = MLP(
            input_dim= input_dim,
            output_dim=output_dim,
            hidden_layers= hidden_layers,
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
        buffer_typ: UER | DQNBuffer,
        update_frequency: int,
        q_net_h_layers: list,
        double_dqn: bool = True
    ):
        self.env = env
        self.double = double_dqn

        assert isinstance(self.env.action_space, Discrete)
        assert isinstance(self.env.observation_space, Box)

        self.buffer_sizes = buffer_sizes
        self.q_net_h_lyaer = q_net_h_layers
        
        self._build_memory()
        self._build_q_net()
        
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.batch_size = batch_size
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.steps = 0

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
            hidden_layers = self.q_net_h_lyaer
        )
        self.target_q_net = deepcopy(self.q_net)

    @torch.no_grad()
    def select_action(self, obs):
        q_values = self.q_net(torch.as_tensor(obs))
        action = q_values.max(dim=0)[1].item()
        return action

    def update(self, logger:Optional[SummaryWriter]):
        try:
            bs, ba, br, bs_, bd = self.memory.sample(self.batch_size)
        except ValueError:
            # Continue when samples less than a batch
            return

        current_q = self.q_net(bs)
        Q_values = torch.gather(input=current_q, dim=2, index=ba)
        
        with torch.no_grad():
            if self.double:
                maxq_action = self.q_net(bs_).max(dim=2)[1].unsqueeze(dim=1)
                target_q = torch.gather(self.target_q_net(bs_), dim=2, index=maxq_action)
            else:
                target_q,_ = self.target_q_net(bs_).max(dim=2)

        y = br + (1-bd)*self.gamma*target_q.squeeze(dim=1)
        loss = F.smooth_l1_loss(y.squeeze(), Q_values.squeeze())
        
        if(logger!=None):
            logger.add_scalar('loss', loss, self.steps)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        if self.steps%self.update_frequency == 0:
            #print("update target")
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, path, name="best.pt"):
        from pathlib import Path
        Path(f"{path}/checkpoints/").mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), f"{path}/checkpoints/{name}")
    
    def load_model(self, path, name="best.pt"):
        self.q_net.load_state_dict(torch.load(f"{path}/checkpoints/{name}"))
        #print("model loaded")
        self.target_q_net = deepcopy(self.q_net)

