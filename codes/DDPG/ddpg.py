from copy import deepcopy
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from .nn import Actor, Critic
from .memory import ExperienceReplay, Transition, PER
from .noise import ActionNoise, AdaptiveParameterNoise


class DDPG:

    def __init__(self,
            actor : Actor,
            critic: Critic,
            actor_optimiser : Callable[[Iterator[Parameter]], Optimizer],
            critic_optimiser: Callable[[Iterator[Parameter]], Optimizer],
            memory: ExperienceReplay,
            batch_size: int,
            discount_factor: float,
            polyak: float,  # Polyak averaging coefficient between 0 and 1
            noise: ActionNoise | AdaptiveParameterNoise | None = None
    ):

        self._actor  = actor
        self._critic = critic
        self._target_actor  = deepcopy(self._actor)
        self._target_critic = deepcopy(self._critic)
        # Freeze target networks with respect to optimisers (only update via polyak averaging)
        self._target_actor.requires_grad_(False)
        self._target_critic.requires_grad_(False)

        self._actor_optimiser  = actor_optimiser(self._actor.parameters())
        self._critic_optimiser = critic_optimiser(self._critic.parameters())

        self.memory = memory
        self._batch_size = batch_size

        self._discount_factor = discount_factor
        self._polyak = polyak
        self._noise = noise


    def update(self):

        try:
            transitions = self.memory.sample(self._batch_size)
        except ValueError:
            # Continue when samples less than a batch
            return

        batch = Transition(*zip(*transitions))
        states      = torch.stack(batch.state)
        actions     = torch.stack(batch.action)
        rewards     = torch.stack(batch.reward)
        next_states = torch.stack(batch.next_state)
        dones       = torch.stack(batch.done)

        TD_targets = rewards + ~dones * self._discount_factor * self._target_critic(next_states, self._target_actor(next_states))
        Qs = self._critic(states, actions)

        Q_loss = F.mse_loss(TD_targets, Qs)
        self._critic_optimiser.zero_grad()
        Q_loss.backward()
        self._critic_optimiser.step()

        # Learn a deterministic policy which gives the action that maximizes Q by gradient ascent
        policy_loss = -self._critic(states, self._actor(states)).mean()
        self._actor_optimiser.zero_grad()
        policy_loss.backward()
        self._actor_optimiser.step()

        # Update frozen target networks by Polyak averaging
        with torch.no_grad():  # stops target param from requesting grad after calc because original param require grad are involved in the calc
            for ϕ, ϕ_targ in zip(self._critic.parameters(), self._target_critic.parameters()):
                ϕ_targ.mul_(self._polyak)
                ϕ_targ.add_( (1.0 - self._polyak) * ϕ )
            for θ, θ_targ in zip(self._actor.parameters(), self._target_actor.parameters()):
                θ_targ.mul_(self._polyak)
                θ_targ.add_( (1.0 - self._polyak) * θ )

        with torch.no_grad():
            match self.memory:
                case PER():
                    TD_errors  = TD_targets - Qs
                    priorities = torch.abs(TD_errors).cpu().numpy()
                    self.memory.update_priorities(priorities)
                case _:
                    pass

    @torch.no_grad()
    def compute_action(self, state: Tensor) -> Tensor:
        action = self._actor(state)
        match self._noise:
            case ActionNoise():
                action += self._noise(action.size(), action.device)
                action.clamp_(-1, 1)  # Output layer of actor network is tanh activated; hence the valid action range is [-1, 1]
            case AdaptiveParameterNoise():
                perturbed_actor  = self._noise.perturb(self._actor)
                perturbed_action = perturbed_actor(state)
                self._noise.adapt(action, perturbed_action)
                action = perturbed_action
            case _:
                pass
        return action
