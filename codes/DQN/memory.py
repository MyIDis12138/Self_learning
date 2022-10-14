import random
from collections import deque, namedtuple
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import torch
from torch import Tensor

from gym import spaces
from utils import get_action_dim, get_obs_shape

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class BaseBuffer(ABC):
    
    def __init__(
        self, 
        buffer_size:int, 
        observation_space:spaces.Space, 
        action_space: spaces.Space,
        envs_num:int = 1
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dim = get_action_dim(self.action_space)
        self.obs_shape = get_obs_shape(self.observation_space)
        self.envs_num = envs_num
        self.full = False
        self.pos = 0

    @abstractmethod
    def add(
        self,
        obs: np.ndarray,
        obs_: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos:List[Dict[str, Any]]
    ):
        ...

    @abstractmethod
    def sample(self, batch_size: int):
        ...

    def __len__(self):
        return self.buffer_size if self.full else self.pos


class UER(BaseBuffer):
    '''
    Uniformly sampled

    In SRS, each subset of k individuals has the same 
    probability of being chosen for the sample as any 
    other subset of k individuals.
    https://en.wikipedia.org/wiki/Simple_random_sample
    '''

    def __init__(self, capacity: int):
        self._buffer = deque(maxlen=capacity)

    def push(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, done: Tensor):
        self._buffer.append( Transition(state, action, next_state, reward, done) )

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buffer, min(batch_size, self._buffer.count()))

class DQNBuffer(BaseBuffer):

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        envs_num: int = 1
    ):
        super(DQNBuffer,self).__init__(buffer_size, observation_space, action_space, envs_num)
        self.observations = np.zeros((self.buffer_size, self.envs_num) + self.obs_shape, dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.envs_num) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.envs_num, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.envs_num), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.envs_num), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        obs_: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos:List[Dict[str, Any]]
    ):
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.envs_num,) + self.obs_shape)
            next_obs = next_obs.reshape((self.envs_num,) + self.obs_shape)

        action = action.reshape((self.envs_num, self.action_dim))
        self.observations[self.pos] = np.array(obs[0]).copy()
        self.next_observations[self.pos] = np.array(obs_[0]).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if(self.full):
            indexes = random.sample(range(0, self.buffer_size), batch_size)
        else:
            indexes = random.sample(range(0, self.pos), batch_size)

        batch_obs = []
        batch_action = []
        batch_reward = []
        batch_done = []
        batch_obs_ = []

        for inx in indexes:
            batch_obs.append(self.observations[inx])
            batch_obs_.append(self.next_observations[inx])
            batch_action.append(self.actions[inx])
            batch_done.append(self.dones[inx])
            batch_reward.append(self.rewards[inx])
        
        batch_obs = np.array(batch_obs)       
        batch_action = np.array(batch_action)
        batch_reward = np.array(batch_reward)
        batch_done = np.array(batch_done)
        batch_obs_ = np.array(batch_obs_)
        
        tensor_obs = torch.as_tensor(batch_obs)
        tensor_obs_ = torch.as_tensor(batch_obs_)
        tensor_action = torch.as_tensor(batch_action)
        tensor_reward = torch.as_tensor(batch_reward)
        tensor_dones = torch.as_tensor(batch_done)

        return tensor_obs, tensor_action, tensor_reward, tensor_obs_, tensor_dones