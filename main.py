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
agent = Agent(state_size=len(env_info.vector_observations[0]),
              action_size=brain.vector_action_space_size, seed=0)


# Train the agent
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        # We will leave a maximum time in here for now
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()


# # ### 3. Train the Agent with DQN
# #
# # Run the code cell below to train the agent from scratch.  You are welcome to amend the supplied values of the parameters in the function, to try to see if you can get better performance!
#
# # In[ ]:
#
#
#
# # plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()
#
#
# # ### 4. Watch a Smart Agent!
# #
# # In the next code cell, you will load the trained weights from file to watch a smart agent!
#
# # In[ ]:
#
#
# # load the weights from file
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
#
# for i in range(5):
#     state = env.reset()
#     for j in range(200):
#         action = agent.act(state)
#         env.render()
#         state, reward, done, _ = env.step(action)
#         if done:
#             break
#
# env.close()
#
#
# # ### 5. Explore
# #
# # In this exercise, you have implemented a DQN agent and demonstrated how to use it to solve an OpenAI Gym environment.  To continue your learning, you are encouraged to complete any (or all!) of the following tasks:
# # - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster.  Once you build intuition for the hyperparameters that work well with this environment, try solving a different OpenAI Gym task with discrete actions!
# # - You may like to implement some improvements such as prioritized experience replay, Double DQN, or Dueling DQN!
# # - Write a blog post explaining the intuition behind the DQN algorithm and demonstrating how to use it to solve an RL environment of your choosing.
