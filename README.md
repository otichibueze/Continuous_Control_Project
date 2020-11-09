# Train A Double-Jointed Arm to move to target locations
For this project, we would train a single and multi agents in a unity envrioment 

**Watch trained agents**

_single agent_

![alt text](https://github.com/otichibueze/p2_continuous_control_project/blob/master/single_agent.gif)


_Sharing experience can accelerate learning_

![alt text](https://github.com/otichibueze/p2_continuous_control_project/blob/master/multi_agent.gif)

![alt text](https://github.com/otichibueze/p2_continuous_control_project/blob/master/multi.gif)

[Source](https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html)


### Enviroment State  Action Reward
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Getting Started
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

#### Version 1: One (1) Agent
- Linux [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit) [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit) [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
    
#### Version 2: Twenty (20) Agents
- Linux [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit) [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit) [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    
 (For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)
    
    
### Learning Algorithm
To get started we need to determine the algorithm best suited for Reacher enviroment which is a continous action space.
Policy based methods are used when an enviroment has infinite states.

**POLICY BASED METHODS**
A policy defines the learning agent's way of behaving at a given time. Roughly speaking, a policy is a mapping from perceived states of the environment to actions to be taken when in those states.
    
**Algorithms Options**
- Promixal Policy Optimization PPO [paper](https://arxiv.org/abs/1707.06347)
- (Continuous/Discrete) Synchronous Advantage Actor Critic (A2C) [paper](https://arxiv.org/abs/1602.01783v2)
- Deep Deterministic Policy Gradient (DDPG) [paper](https://arxiv.org/pdf/1509.02971.pdf)
    
####  Deep Deterministic Policy Gradient
we will implement Deep Deterministic Policy Gradient (DDPG) for this project. This works with Actor-Critic methods.
Actor critic methods are at the intersection of value-based methods such as DQN and policy-based methods such as reinforce. If a deep reinforcement learning agent uses a deep neural network to approximate a  value function the agent is said to be value-based, if an agent uses a deep neural network to approximate a policy the agent is said to be policy based. 
Actor-critic methods combine these two approaches in order to accelerate the learning process. 

#### Hyper Parameters Used
- BUFFER_SIZE = int(2e6)  
- BATCH_SIZE = 128      
- GAMMA = 0.99
- TAU = 1e-3
- LR_ACTOR = 1e-4      
- LR_CRITIC = 1e-4        
- WEIGHT_DECAY = 0
- UPDATE_EVERY = 10 
- Size of hidden layers = 256, 128, 64


**Gradient Clipping**
We had troubles getting the agent to learn because the weight from the critic model is becoming quite large after a few episodes of training making updates too large to allow the model to learn. Using gradient clipping when training the critic network we can combat this problem code below

```
self.critic_optimizer.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
self.critic_optimizer.step()
```
    
**Batch Normalization**
This is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.
 
Similar to the exploding gradient issue mentioned above, running computations on large input values and model parameters can inhibit learning. Batch normalization addresses this problem by scaling the features to be within the same range throughout the model and across different environments and units. In additional to normalizing each dimension to have unit mean and variance, the range of values is often much smaller, typically between 0 and 1. We implementeed on first layer of fully connected layers of both actor and critic models
    
    
**Experience Replay**
Experience replay allows the RL agent to learn from past experience.

As with DQN in the previous project, DDPG also utilizes a replay buffer to gather experiences from each agent. Each experience is stored in a replay buffer as the agent interacts with the environment. In this project, there is one central replay buffer utilized by all 20 agents, therefore allowing agents to learn from each others' experiences.

The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). Each agent samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive algorithm could otherwise become biased by correlations between sequential experience tuples.

**Updates Per Time Step**
The number rof updates per time step affects performance of agent, So instead of updating the actor and critic networks every timestep, we update the networks after every 10 time steps.
    
**Sharing Experience**
In the second version of the project environment, there are 20 identical copies of the agent. It has been shown that having multiple copies of the same agent [sharing experience can accelerate learning](https://ai.googleblog.com/2016/10/how-robots-can-acquire-new-skills-from.html), and you'll discover this for yourself when solving the project!
    
### Run Experiments
_Single Agent_

![alt text](https://github.com/otichibueze/p2_continuous_control_project/blob/master/single.png)


_Multiple Agents_

![alt text](https://github.com/otichibueze/p2_continuous_control_project/blob/master/multi.png)



    
### Future Improvements 
- **Add prioritized experience replay**  Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated with the magnitude of error. This can improve learning by increasing the probability that rare and important experience vectors are sampled.


**Experiment with other algorithms like**
- PPO [paper](https://arxiv.org/abs/1707.06347) 
- (A2C) [paper](https://arxiv.org/abs/1602.01783v2)


### Optional Challenge: Crawl

_Teach a creature with four legs to walk forward without falling_
![alt text](https://video.udacity-data.com/topher/2018/August/5b633811_crawler/crawler.png)

**Download the Unity Environment**
To solve this harder task, you'll need to download a new Unity environment. You need only select the environment that matches your operating system:
- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)
