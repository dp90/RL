#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:16:43 2020

@author: david
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.tensor as tnsr

import gym
import numpy as np
import matplotlib.pyplot as plt
import random

torch.set_default_dtype(torch.float64)

def plot_running_average(rewards):
    N = len(rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(rewards[max(0, t-100) : (t+1)])
    plt.figure()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()
    return

class ActionValue(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, output_dim)
        
    def forward(self, state):
        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)
    
class Policy(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, output_dim)
    
    def forward(self, state):
        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))
        return torch.tanh(self.fc3(h2))

class Experience(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []
    
    def add(self, transition):
        if len(self.memory) >= self.max_size:
            pos = random.randrange(self.max_size)
            self.memory[pos] = transition
        else:
            self.memory.append(transition)
    
    def sample_batch(self, batch_size):
        if batch_size < len(self.memory):
            batch = random.sample(self.memory, batch_size)
        else:
            batch = self.memory
        
        s, a, r, s_, d = tuple(zip(*batch))
        return tnsr(s), tnsr(a), tnsr(r), tnsr(s_), tnsr(d)*1

def update_target_network(targ_net, pred_net, TAU):
	for targ_param, pred_param in zip(targ_net.parameters(), pred_net.parameters()):
		targ_param.data.copy_(TAU*pred_param.data + (1-TAU)*targ_param.data)

env = gym.make('Pendulum-v0') # state = (x, y, theta), action = (torque)

actor_input_dim = env.observation_space.shape[0]
critic_input_dim = env.observation_space.shape[0] + 1
h1_dim = 16
h2_dim = 16
output_dim = 1

# Initiate actor and critic networks
mu_pred = Policy(actor_input_dim, h1_dim, h2_dim, output_dim)
mu_targ = Policy(actor_input_dim, h1_dim, h2_dim, output_dim)
Q_pred = ActionValue(critic_input_dim, h1_dim, h2_dim, output_dim)
Q_targ = ActionValue(critic_input_dim, h1_dim, h2_dim, output_dim)
mu_targ.load_state_dict(mu_pred.state_dict())
Q_targ.load_state_dict(Q_pred.state_dict())

lr_mu = 1e-3
lr_Q = 1e-3
optim_mu = optim.Adam(mu_pred.parameters(), lr=lr_mu)
optim_Q = optim.Adam(Q_pred.parameters(), lr=lr_Q)

max_exp_size = 100000
experience = Experience(max_exp_size)
batch_size = 128

GAMMA = 0.99
TAU = 0.01
SIGMA = 0.1

n_episodes = 200
reward_history = []

for i in range(n_episodes):
    done = False
    ep_rewards = 0
    
    state = env.reset()
    
    while not done:
        # Add transition to experience
        action = mu_pred(tnsr(state)).detach().numpy() + np.random.randn(1)*SIGMA
        state_, reward, done, _ = env.step(action)
        ep_rewards += reward
        
        experience.add((state, action, reward, state_, done))
        state = state_
        
        # Update actor and critic networks with mini-batch from experience
        s, a, r, s_, d = experience.sample_batch(batch_size) # returns tensors

        a_ = mu_targ(s_).detach()
        val_targ = r + GAMMA*(1-d)*Q_targ(torch.cat((s_, a_), dim=1)).detach().squeeze()
        val_pred = Q_pred(torch.cat((s, a), dim=1)).squeeze()
        
        Q_loss = F.mse_loss(val_targ, val_pred)
        optim_Q.zero_grad()
        Q_loss.backward()
        optim_Q.step()
        
        val_pred = Q_pred(torch.cat((s, mu_pred(s)), dim=1)).squeeze()
        
        mu_loss = -torch.mean(val_pred)
        optim_mu.zero_grad()
        mu_loss.backward()
        optim_mu.step()
        update_target_network(mu_targ, mu_pred, TAU)
        update_target_network(Q_targ, Q_pred, TAU)
                
    reward_history.append(ep_rewards)
    if i % 5 == 0:
        print()
        print("Episode: %d" % (i))
        print("Reward: %.2f" % (ep_rewards))
        print("mu loss: %.2f" % (mu_loss.detach().numpy()))

env.close()

# Plot learning curve
plot_running_average(reward_history)






