# RL - Papers2Python

This repository contains the several successful attempts to implement algorithms from scientific papers in Python from scratch, tested in the OpenAI gym. Currently included:
- Deep Q Network (DQN)
- Proximal Policy Optimization (PPO)
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG (TD3)

## Deep Q Network
The deep Q network algorithm, as described [here](https://arxiv.org/abs/1312.5602), was tested with the cart-pole environment. While the learning curve below shows a relatively stable increase in the rewards the algorithm is able to achieve for each episode, the results are highly dependent on the various hyperparameters. 

![alt text](https://github.com/dp90/ReinforcementLearning/blob/master/Images/DQN_learningCurve.png "DQN learning curve")

## Proximal Policy Optimization
The proximal policy optimization algorithm, as described [here](https://arxiv.org/abs/1707.06347), was tested with the cart-pole environment. The learning curve below shows a steadily increasing reward as the algorithm trains, which is precisely its aim: steady small updates. 

![alt text](https://github.com/dp90/ReinforcementLearning/blob/master/Images/PPO_learningCurve.png "PPO learning curve")

## Deep Deterministic Policy Gradient
The deep deterministic policy gradient algorithm, as described [here](https://arxiv.org/abs/1509.02971), was tested with the inverted pendulum environment. While DDPG was one of the first algorithms to combine policy gradient methods in a continuous action space, it is not overly robust. Running the script multiple times, or slightly changing the hyperparameters can produce rather different looking learning curves from the one shown below. 

![alt text](https://github.com/dp90/ReinforcementLearning/blob/master/Images/DDPG_learningCurve.png "DDPG learning curve")

## Twin Delayed DDPG
The twin delayed DDPG algorithm, as described [here](https://arxiv.org/abs/1802.09477), was tested with the inverted pendulum environment. It improves the robustness of the DDPG by i) learning 2 Q functions and using the smallest result of the two to reduce value overestimating ii) delaying updates of the policy and target networks iii) adding noise to the target policy's actions to prevent overfitting to possibly incorrect values. A more detailed [explanation](https://spinningup.openai.com/en/latest/algorithms/td3.html) is provided by OpenAI. A resulting learning curve is shown below shows. 

![alt text](https://github.com/dp90/ReinforcementLearning/blob/master/Images/TD3_learningCurve.png "TD3 learning curve")
