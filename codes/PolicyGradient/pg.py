from operator import le
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=36):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class PolicyGradient:

    def __init__(self, n_states, cfg) -> None:
        self.gamma = cfg.gamma
        self.policy_net = MLP(n_states, hidden_dim= cfg.hidden_dim)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters, lr = cfg.lr)
        self.batch_size = cfg.batch_size

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state)
        m = Bernoulli(probs)
        action = m.sample()
        action = action.data.numpy().astype(int)[0]
        return action
    
    def update(self, reward_pool, state_pool, action_pool):
        running_add = 0

        # accumulate reward with discount factor gamma
        for i in reversed(range(len(reward_pool))):
            if(reward_pool[i]) == 0:
                running_add = 0
            else:
                running_add = running_add*self.gamma + reward_pool[i]
                reward_pool[i] = running_add
        
        # Normoalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
        

        # Gradient Desent
        self.optimizer.zero_grad()

        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward
            print(loss)
            loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'pg_checkpoint.pt')
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'pg_checkpoint.pt'))
