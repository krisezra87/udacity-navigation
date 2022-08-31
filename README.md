# Project 1: Navigation

## Introduction

This project is the solution to a unity-based, banana-scavenging task using deep Q-networks (DQN).

Within this environment, a reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
The agent's goal is to avoid blue bananas and collect as many yellow bananas as possible.

### The Environment
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.
Given this information, the agent learns which actions are best.  The action space is discrete and is characterized as follows:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Downloading the environment and dependencies

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the root of this repository and unzip (or decompress) the file.

3. Change line 14 of `main.py` accordingly if you are using something that is not 64-bit linux (but you should be ;)).

4. Other necessary install instructions for dependencies can be found in the DRLND github repo [located here](https://github.com/udacity/Value-based-methods#dependencies).

## Repository Anatomy

### main.py
When all dependencies are installed, the agent itself can be initialized and trained by running `python main.py`.  When the agent is successfully trained, it will save the parameters charactierizing neural networks for the agent behavior in `./checkpoint.pth` and also present a plot of agent score versus episode.

### demo.py
As long as `./checkpoint.pth` is present, executing `python demo.py` will show the trained agent in action, collecting its bananas!

### dqn_agent.py
This file contains the code for the learning agent itself, complete with the training logic.  A more detailed overview can be found in [the report](Report.md)

### model.py
This file contains the pytorch-based neural network that is used to determine agent actions for a given state.  Again, a detailed overview can be found in [the report](Report.md)
