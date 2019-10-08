# Q learning

This work is a tensorflow based reimplementation of the paper titled **Playing Atari with Deep Reinforcement Learning** which can be found [here](https://arxiv.org/abs/1312.5602).  

OpenAI gym provides a number of different environments for reinforcement learning research. For this particular Q-Learning task, I used two such environments whose names are:
1. CartPole-v0
2. MountainCar-v0

## Usage
To initiate training, and automatically set the hyperparameters based on the environment, please type:
```
python DQN_Implementation.py --env <environment name>
```

### System Settings
OS - Ubuntu 14.04

GPU - NVIDIA GTX1080Ti

Python - 3.4

Tensorflow 1.4.0

For more dependencies, please check the requirements.txt file. 
