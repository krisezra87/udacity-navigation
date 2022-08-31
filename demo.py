# Deep Q-Network (DQN) approach to training banana-collecting agent
# ---

import torch
import numpy as np
from collections import deque
from dqn_agent import Agent
from unityagents import UnityEnvironment


# First configure the environment
# NOTE: I have configured for LINUX.
env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Grab the environment info as well to set up the agent
env_info = env.reset(train_mode=True)[brain_name]

# Set up the agent generically for state and action sizes.  Can't ever be TOO
# portable!
agent = Agent(state_size=len(env_info.vector_observations[0]),
              action_size=brain.vector_action_space_size, seed=0)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = agent.act(state, 0) # Set epsilon to zero.  We have a policy we like
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break

print("Score: {}".format(score))

env.close()
