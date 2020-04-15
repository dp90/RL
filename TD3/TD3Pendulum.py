#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:12:17 2020

@author: david
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    plt.ylabel("Rewards (running average)")
    plt.xlabel("Episodes")
    plt.show()
    return

class Actor(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, output_dim)
    
    def forward(self, state):
        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))
        return torch.tanh(self.fc3(h2))

class Critic(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, output_dim)
    
    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        h1 = F.relu(self.fc1(state_action))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

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
        return torch.tensor(s), torch.tensor(a).double(), torch.tensor(r),\
                torch.tensor(s_), torch.tensor(d)*1

def update_target_network(targ_net, pred_net, TAU):
	for targ_param, pred_param in zip(targ_net.parameters(), pred_net.parameters()):
		targ_param.data.copy_(TAU*pred_param.data + (1-TAU)*targ_param.data)

env = gym.make('Pendulum-v0')
DIM_A_SPACE = env.action_space.shape[0]
DIM_S_SPACE = env.observation_space.shape[0]

# Create actor and critic networks
H1_DIM = 16
H2_DIM = 16
actor = Actor(DIM_S_SPACE, H1_DIM, H2_DIM, DIM_A_SPACE)
critic1 = Critic(DIM_S_SPACE+DIM_A_SPACE, H1_DIM, H2_DIM, 1)
critic2 = Critic(DIM_S_SPACE+DIM_A_SPACE, H1_DIM, H2_DIM, 1)
optim_actor = optim.Adam(actor.parameters())
optim_critic1 = optim.Adam(critic1.parameters())
optim_critic2 = optim.Adam(critic2.parameters())

# Create the targets of the actor and critic networks
actor_targ = Actor(DIM_S_SPACE, H1_DIM, H2_DIM, DIM_A_SPACE)
critic1_targ = Critic(DIM_S_SPACE+DIM_A_SPACE, H1_DIM, H2_DIM, 1)
critic2_targ = Critic(DIM_S_SPACE+DIM_A_SPACE, H1_DIM, H2_DIM, 1)

# Assign weights of actor and critics to their target networks
actor_targ.load_state_dict(actor.state_dict())
critic1_targ.load_state_dict(critic1.state_dict())
critic2_targ.load_state_dict(critic2.state_dict())

max_exp_size = 10000
experience = Experience(max_exp_size)
batch_size = 64
n_updates_per_step = 4
min_exp_size = batch_size * n_updates_per_step

GAMMA = 0.99
TAU = 0.01 # weight given to updated network in average with target network
SIGMA = 0.1
C = 2.5*SIGMA
ACTOR_DELAY = 2

n_episodes = 100
reward_hist = []

while len(experience.memory) < min_exp_size:
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        state_, reward, done, _ = env.step(action)
        experience.add((state, action, reward, state_, done))
        state = state_

for i in range(n_episodes):
    done = False
    ep_rewards = 0
    
    state = env.reset()
    
    while not done:
        action = actor(torch.tensor(state)).detach().numpy() \
                + np.clip(np.random.randn(1)*SIGMA, -C, C, dtype=np.float64)
        state_, reward, done, _ = env.step(action)
        ep_rewards += reward
        
        experience.add((state, action, reward, state_, done))
        state = state_
        
        # Update actor and critics
        for j in range(n_updates_per_step):
            s, a, r, s_, d = experience.sample_batch(batch_size)
            
            a_ = actor_targ(s_).detach() + torch.clamp(torch.randn(1)*SIGMA, -C, C)
            min_Q_targ = torch.min(critic1_targ(s_, a_), critic2_targ(s_, a_)).squeeze().detach()
            V_targ = r + (1-d)*GAMMA*min_Q_targ
            
            critic1_loss = F.mse_loss(critic1(s, a).squeeze(), V_targ)
            optim_critic1.zero_grad()
            critic1_loss.backward()
            optim_critic1.step()
            
            critic2_loss = F.mse_loss(critic2(s, a).squeeze(), V_targ)
            optim_critic2.zero_grad()
            critic2_loss.backward()
            optim_critic2.step()
            
            if j % ACTOR_DELAY == 0:
                actor_loss = -torch.mean(critic1(s, actor(s)))
            
                optim_actor.zero_grad()
                actor_loss.backward()
                optim_actor.step()
                
                update_target_network(critic1_targ, critic1, TAU)
                update_target_network(critic2_targ, critic2, TAU)
                update_target_network(actor_targ, actor, TAU)
    
    if i % 5 == 0:
        print("Episode %d" % (i+1))
        print("Reward:     %.2f" % (ep_rewards))
        print("Actor loss: %.2f" % (actor_loss))
        print(" ")
    
    reward_hist.append(ep_rewards)

# End of training: plot rewards
plot_running_average(reward_hist)



