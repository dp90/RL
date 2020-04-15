# RL - Papers2Python

This repository contains the several successful attempts to implement algorithms from scientific papers in Python from scratch, tested in the OpenAI gym. Currently included:
- Deep Q Network (DQN)
- Proximal Policy Optimization (PPO)
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG (TD3)

## Deep Q learning
The deep Q learning algorithm, as described [here](https://arxiv.org/abs/1312.5602), was tested on the CartPole example. While the learning curve below shows a relatively stable increase in the rewards the algorithm is able to achieve for each episode, the results are highly dependent on the various hyperparameters. 

![alt text](https://github.com/dp90/ReinforcementLearning/blob/master/Images/DQN_learningCurve.png "DQN learning curve")
