from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils import MLP
from memory import DQNBuffer
import gym
from gym.spaces import Discrete, Box

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
        update_frequency: int,
        q_net_h_layers: list
    ):
        self.env = env

        assert isinstance(self.env.action_space, Discrete)
        assert isinstance(self.env.observation_space, Box)


        self.buffer_sizes = buffer_sizes
        
        self.q_net_h_lyaer = q_net_h_layers
        self.batch_size = batch_size
        self._build_memory()
        self._build_q_net()
        
        self.gamma = gamma
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
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

    def update(self):
        try:
            bs, ba, br, bs_, bd = self.memory.sample(self.batch_size)
        except ValueError:
            # Continue when samples less than a batch
            return

        
        # Q(s_t, a_t)
        current_q = self.q_net(bs)
        Q_values = torch.gather(input=current_q, dim=2, index=ba)

        # Q(s_t+1, argmax(Q(s_t,a_t)))
        bs_act_ = self.q_net(bs_).max(dim=2)[1].unsqueeze(dim=1)
        with torch.no_grad():
            target_q = self.target_q_net(bs_)
            target_q = torch.gather(input=target_q, dim=2, index=bs_act_)
        
        # y = r_t + gamma*Q(s_t+1, argmax(Q(s_t,a_t)))
        y = br + (1-bd)*self.gamma*target_q
        
        # loss = [y - Q(s_t,a_t)]^2
        loss = F.smooth_l1_loss(y, Q_values)
        
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