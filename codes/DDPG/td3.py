from copy import deepcopy
from typing import Callable, Iterator
from itertools import count

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from .nn import Actor, Critic
from .memory import ExperienceReplay, Transition
from .noise import Gaussian


class TD3:
    '''Twin-Delayed DDPG'''

    def __init__(self,
            actor : Actor,
            critic: Critic,
            actor_optimiser : Callable[[Iterator[Parameter]], Optimizer],
            critic_optimiser: Callable[[Iterator[Parameter]], Optimizer],
            memory: ExperienceReplay,
            batch_size: int,
            discount_factor: float,
            polyak: float,
            noise: Gaussian,
            clip_bound: float,
            stddev: float,
            critic_num: int = 2,
            policy_delay: int = 2
    ):

        self._actor = actor
        self._critics = [deepcopy(critic) for _ in range(critic_num)]
        self._target_actor = deepcopy(self._actor)
        self._target_critics = deepcopy(self._critics)
        # Freeze target networks with respect to optimisers (only update via polyak averaging)
        self._target_actor.requires_grad_(False)
        [critic.requires_grad_(False) for critic in self._target_critics]

        self._actor_optimiser = actor_optimiser(self._actor.parameters())
        self._critic_optimisers = [critic_optimiser(critic.parameters()) for critic in self._critics]

        self.memory = memory
        self._batch_size = batch_size

        self._discount_factor = discount_factor
        self._polyak = polyak
        self._noise = noise
        self._clip_bound = clip_bound
        self._stddev = stddev
        self._policy_delay = policy_delay

        self._count = count(start=1, step=1)

    def update(self):

        try:
            transitions = self.memory.sample(self._batch_size)
        except ValueError:
            return

        batch = Transition(*zip(*transitions))
        states      = torch.stack(batch.state)
        actions     = torch.stack(batch.action)
        rewards     = torch.stack(batch.reward)
        next_states = torch.stack(batch.next_state)
        dones       = torch.stack(batch.done)

        # Compute target actions
        target_actions = self._target_actor(next_states)
        # Add clipped noise
        target_actions += target_actions.clone().normal_(0, self._stddev).clamp_(-self._clip_bound, self._clip_bound)
        # Target actions are clipped to lie in valid action range
        target_actions.clamp_(-1, 1)  # Output layer of actor network is tanh activated; hence the valid action range is [-1, 1]

        TD_targets = rewards + ~dones * self._discount_factor * torch.min(*[target_critic(next_states, target_actions) for target_critic in self._target_critics])
        ls_Qs = [critic(states, actions) for critic in self._critics]

        Q_loss = torch.add(*[F.mse_loss(TD_targets, Qs) for Qs in ls_Qs])
        [critic_optimiser.zero_grad() for critic_optimiser in self._critic_optimisers]
        Q_loss.backward()
        [critic_optimiser.step() for critic_optimiser in self._critic_optimisers]

        if next(self._count) % self._policy_delay == 0:

            # Learn a deterministic policy which gives the action that maximizes Q by gradient ascent
            policy_loss = -self._critics[0](states, self._actor(states)).mean()
            self._actor_optimiser.zero_grad()
            policy_loss.backward()
            self._actor_optimiser.step()

            # Update frozen target networks by Polyak averaging
            with torch.no_grad():  # stops target param from requesting grad after calc because original param require grad are involved in the calc
                for critic, target_critic in zip(self._critics, self._target_critics):
                    for ϕ, ϕ_targ in zip(critic.parameters(), target_critic.parameters()):
                        ϕ_targ.mul_(self._polyak)
                        ϕ_targ.add_( (1.0 - self._polyak) * ϕ )
                for θ, θ_targ in zip(self._actor.parameters(), self._target_actor.parameters()):
                    θ_targ.mul_(self._polyak)
                    θ_targ.add_( (1.0 - self._polyak) * θ )

    @torch.no_grad()
    def compute_action(self, state: Tensor) -> Tensor:
        action = self._actor(state)
        match self._noise:
            case Gaussian():
                action += self._noise(action.size(), action.device)
                action.clamp_(-1, 1)  # Output layer of actor network is tanh activated; hence the valid action range is [-1, 1]
            case _:
                pass
        return action
