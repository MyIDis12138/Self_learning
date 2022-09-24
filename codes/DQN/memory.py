import warnings
from typing import List
from abc import ABC, abstractmethod
from collections import namedtuple

from gym import spaces
import numpy as np

from common.memory_utils import get_action_dim, get_obs_shape

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,                    
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False

    
    def size(self) -> int:
        if self.full:
            return self.buffer_size
        return self.pos

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    @abstractmethod
    def push(
            self, 
            obs: np.ndarray, 
            next_obs:np.ndarray,
            action: np.ndarray, 
            reward: np.ndarray, 
            done: np.ndarray
    ):
        ...

    @abstractmethod
    def sample(
        self, 
        batch_size: int) -> List[Transition]:
        ...


class ReplayBuffer(BaseBuffer):
    def __init__(
        self, buffer_size: int, 
        observation_space: spaces.Space, 
        action_space: spaces.Space,

    ):
        super().__init__(buffer_size, observation_space, action_space)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.observations = np.zeros((self.buffer_size) + self.obs_shape, dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )
    
    def push(
        self, 
        obs: np.ndarray, 
        next_obs:np.ndarray,
        action: np.ndarray, 
        reward: np.ndarray, 
        done: np.ndarray
    ):
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
