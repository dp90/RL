#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:35:52 2020

@author: david
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:01:41 2020

@author: david
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import time

torch.set_default_dtype(torch.float64)

class Policy(nn.Module):
    def __init__(self, input_dim, h1_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, output_dim)
        
    def forward(self, state):
        state = torch.Tensor(state)
        h1 = F.relu(self.fc1(state))
        out = self.fc2(h1)
        return F.softmax(out, dim=-1)

class Value(nn.Module):
    def __init__(self, input_dim, h1_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, output_dim)
    
    def forward(self, state):
        state = torch.Tensor(state)
        h1 = F.relu(self.fc1(state))
        return self.fc2(h1)

class Experience(object):
    def __init__(self):
        self.actions = torch.tensor([], dtype=torch.long)
        self.states = torch.tensor([])
        self.pi_olds = torch.tensor([])
        self.G = torch.tensor([]) # returns
        self.A = torch.tensor([]) # advantages
    
    def add(self, actions, states, pi_olds, G, A):
        self.actions = torch.cat((self.actions, actions), dim=0)
        self.states = torch.cat((self.states, states), dim=0)
        self.pi_olds = torch.cat((self.pi_olds, pi_olds), dim=0)
        self.G = torch.cat((self.G, G), dim=0)
        self.A = torch.cat((self.A, A), dim=0)
    
    def sample(self, batch_size):
        mask = torch.randperm(len(self.actions))[:batch_size]
        action_batch = self.actions[mask]
        state_batch = self.states[mask]
        pi_old_batch = self.pi_olds[mask]
        G_batch = self.G[mask]
        A_batch = self.A[mask]
        return action_batch, state_batch, pi_old_batch, G_batch, A_batch
        
    def __len__(self):
        return len(self.actions)
        
def plot_running_average(x, window=100):
    N = len(x)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(x[max(0, t-window) : (t+1)])
    plt.figure()
    plt.plot(running_avg)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards (running average)")
    plt.show()
    return

env = gym.make('CartPole-v1')
n_episodes = 32
n_iterations = 25
reward_history = []

input_dim = env.observation_space.shape[0]
h1_dim = 32
output_dim = env.action_space.n
policy = Policy(input_dim, h1_dim, output_dim)
value = Value(input_dim, h1_dim, 1)

experience = Experience()
batch_size = 16

lr_pol = 1e-3
lr_val = 1e-3
optim_pol = optim.Adam(policy.parameters(), lr=lr_pol)
optim_val = optim.Adam(value.parameters(), lr=lr_val)

GAMMA = 0.99
LAMBDA = 0.9
EPS_CLIP = 0.2

for iteration_i in range(n_iterations):
    iter_tic = time.time()
    
    for episode in range(n_episodes):
        done = False
        state = env.reset()
        V = value(state)
        
        states = []
        actions = []
        rewards = []
        pi_olds = []
        deltas = []
        
        while not done: # something with maximum number of steps
            a_probs_old = policy(state)
            action = torch.distributions.Categorical(a_probs_old).sample()
            
            pi_old = a_probs_old[action]
            
            state_, reward, done, _ = env.step(action.numpy())
            
            with torch.no_grad():
                V_ = value(state_)
                delta = reward + GAMMA*V_*(1-int(done)) - V
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            pi_olds.append(pi_old)
            deltas.append(delta)
            
            state = state_
            V = V_
        
        T = len(deltas)
        advantage = torch.zeros(T, requires_grad=False)
        returns = torch.zeros(T, requires_grad=False)
        for t in reversed(range(T)):
            advantage[t] = deltas[t] + (0 if t==T-1 else GAMMA*LAMBDA*advantage[t+1])
            returns[t] = rewards[t] + (0 if t == T-1 else GAMMA*returns[t+1])
        
        actions = torch.tensor(actions, requires_grad=False)
        states = torch.tensor(states, requires_grad=False)
        pi_olds = torch.tensor(pi_olds, requires_grad=False)
        experience.add(actions, states, pi_olds, returns, advantage)
        reward_history.append(np.sum(np.array(rewards)))
        
    # Update the value and policy functions
    n_minibatches = len(experience) // batch_size
    for i in range(n_minibatches):
        actions, states, pi_olds, G, A = experience.sample(batch_size)
        
        action_probs = policy(states)
        pi_news = torch.gather(action_probs, 1, actions.unsqueeze(dim=1))
        
        r = torch.squeeze(pi_news) / pi_olds

        loss_clip = -torch.mean(torch.min(r*A, torch.clamp(r, 1-EPS_CLIP, 1+EPS_CLIP)*A))
        
        val_predct = value(states)
        val_target = G
        loss_value = torch.mean((val_target - val_predct)**2)
        
        # Update parameters of policy and value functions
        optim_pol.zero_grad()
        optim_val.zero_grad()
        (loss_clip + loss_value).backward()
        optim_pol.step()
        optim_val.step()
    
    print("Iteration %d took %.2f seconds" % (iteration_i + 1, time.time()-iter_tic))

plot_running_average(reward_history)