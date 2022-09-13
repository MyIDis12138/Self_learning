import random
from collections import deque, namedtuple
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from torch import Tensor


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ExperienceReplay(ABC):

    @abstractmethod
    def push(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, done: Tensor):
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> List[Transition]:
        ...


class UER(ExperienceReplay):
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
        return random.sample(self._buffer, batch_size)


class PER(ExperienceReplay):
    '''Prioritised'''

    def __init__(self, capacity: int, alpha: float, epsilon : float = 0.01):
        self._transitions = [None for _ in range(capacity)]
        self._sum_tree = np.zeros(capacity * 2 - 1)

        def cycling_idx():
            idx = 0
            while True:
                yield idx
                idx += 1
                if idx == capacity:
                    idx = 0
        self._transition_idx = cycling_idx()
        self._idx_bias = len(self._sum_tree) - capacity

        self._alpha = alpha
        self._epsilon = epsilon

        self._maximal_priority = 1.0
        self._sampled_indices = None

    def push(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, done: Tensor):
        transition_idx = next(self._transition_idx)
        self._transitions[transition_idx] = Transition(state, action, next_state, reward, done)
        self._update_priority(transition_idx, self._maximal_priority)

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size > len(self._transitions) - self._transitions.count(None):
            raise ValueError

        def retrieve_index(query_value, node = 0) -> int:
            left  = node * 2 + 1
            right = left + 1

            try:
                left_value = self._sum_tree[left]
            except IndexError:
                return node - self._idx_bias

            if query_value < left_value or np.isclose(query_value, left_value, rtol=1e-3):
                return retrieve_index(query_value, left)
            else:
                return retrieve_index(query_value - left_value, right)

        bounds = np.linspace(0, self._sum_tree[0], batch_size + 1)
        self._sampled_indices = [retrieve_index( random.uniform(low, high) ) for low, high in zip(bounds, bounds[1:])]

        return [self._transitions[idx] for idx in self._sampled_indices]

    def _update_priority(self, transition_idx: int, priority: float):
        node_idx = transition_idx + self._idx_bias
        change   = (priority + self._epsilon) ** self._alpha - self._sum_tree[node_idx]

        while True:
            self._sum_tree[node_idx] += change
            if node_idx == 0:  # reached root node
                break
            node_idx = (node_idx - 1) // 2  # moves to parent node

        self._maximal_priority = max(self._maximal_priority, priority)

    def update_priorities(self, priorities):
        # assert len(self._sampled_indices) == len(priorities)
        for idx, priority in zip(self._sampled_indices, priorities):
            # assert priority > 0
            self._update_priority(idx, priority)
