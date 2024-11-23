import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

from config import CONFIG



class Policy(nn.Module):
    def __init__(self, s_size, h_size, a_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.norm = nn.LayerNorm(h_size)
        # self.fc2 = nn.Linear(h_size, h_size)
        self.fc_mean = nn.Linear(h_size, a_size)
        self.fc_log_std = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        # x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-10, max=10)  # 对log_std进行裁剪
        std = torch.exp(log_std)
        return mean, std

    def act(self, state):
        mean, std = self.forward(state)
        
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, scale_tril=torch.tril(torch.diag_embed(std)))
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    

class RL():
    def __init__(self, env, policy, optimizer):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer

        self.gamma = CONFIG['gamma']

        self.batch_size = CONFIG['batch_size']

        self.log_prob_list = []
        self.reward_list = []

    def get_one_data(self, log_prob, reward):

        if reward is None or len(self.reward_list) >= self.batch_size:
            self.reinforce()
            self.obs_list = []
            self.log_prob_list = []
            self.reward_list = []
        
        else:
            self.log_prob_list.append(log_prob)
            self.reward_list.append(reward)

    def reinforce(self):
        # Help us to calculate the score during the training

        scores = sum(self.reward_list)
        print('Episode Average Score: {:.2f}'.format(scores))

        returns = deque(maxlen=len(self.reward_list))
        n_steps = len(self.reward_list)
        
        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( self.gamma * disc_return_t + self.reward_list[t]   )

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(self.log_prob_list, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.stack(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return scores 