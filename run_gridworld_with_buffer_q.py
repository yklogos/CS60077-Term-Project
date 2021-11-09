###MODIFIED###
#!/usr/bin/env python3

# TODO:
#     1. verify results
#     2. find a way to remove unecessary actions
#     3. implement PEB
#     4. find more efficient implementation of LRU deque


import copy
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from agents import TabularBufferQAgent
from replay import UniformReplayBuffer, CombinedReplayBuffer
from common import get_running_mean
from envs import get_grid_world

# for reproducablity
SEED = 0xc0ffee
np.random.seed(SEED)

# init enviornment variables
ENV_NAME = "MiniGrid-Empty-16x16-v0"   # ['MiniGrid-Empty-16x16-v0', 'MiniGrid-FourRooms-v0']
env = get_grid_world(env_name=ENV_NAME,seed=SEED)
if 'env' in vars(env).keys():
    env = env.env
NUM_STATES = len(env.grid.grid)
ACTIONS = list(env.actions)
NUM_ACTIONS = len(ACTIONS)
ENV_WIDTH = env.width

# init training variables
ALPHA = 0.1
BATCH_SIZE = 10
EPSILON = 0.1
GAMMA = 1
MAX_STEPS = None
NB_EPISODES = 100
REPLAY_BUFFER_SIZES = [100, 1000, 10000, 100000, 1000000]
REPLAY_BUFFER_TYPE = 'combined'  # ['combined', 'uniform']

def pos_to_state(pos):  # pos: (row, col)
    return pos[0]*ENV_WIDTH + pos[1]

start_q_network = np.zeros((NUM_STATES, NUM_ACTIONS)) # init all values to 0
# # init q_network value of terminal states to 0
# for i,cell in enumerate(env.env.grid.grid):  
#     if cell is None:
#         continue
#     if cell.type == "goal":
#         start_q_network[pos_to_state(cell.init_pos)] = np.zeros(NUM_ACTIONS)

replay_buffer_dict = {'combined': CombinedReplayBuffer, 'uniform': UniformReplayBuffer}

for replay_buffer_size in REPLAY_BUFFER_SIZES:
# for replay_buffer_size in REPLAY_BUFFER_SIZES[:1]:

    env = get_grid_world(max_steps=MAX_STEPS, seed=SEED)
    q_network = copy.deepcopy(start_q_network)
    replay_buffer = replay_buffer_dict[REPLAY_BUFFER_TYPE](replay_buffer_size)
    agent = TabularBufferQAgent(q_network, replay_buffer, range(NUM_STATES), env.action_space, alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA, batch_size=BATCH_SIZE)

    epi_returns = []

    for i in range(NB_EPISODES):
#     for i in range(1):

        obs = env.reset()
        state = pos_to_state(env.agent_pos)
        epi_return = 0
        done = False

        while not done:
#         for i in range(BATCH_SIZE+1):

            if MAX_STEPS:
                if not info['timeout']:
                    break
            action = agent.get_action(state, is_table=True)
            next_obs, rew, done, info = env.step(action)

#             # print positive reward attributes 
#             if rew>0:
#                 temp=3
#                 print('\n'*temp)
#                 print(f"run_gridworld_with_buffer_q.py state, rew, done, info: {state, rew, done, info}")
#                 print('\n'*temp)

            rew -= 1   # -1 reward at each step
            next_state = pos_to_state(env.agent_pos)
            transition = (state, action, next_state, rew, done) 
            replay_buffer.append(transition)

#             print(f"run_gridworld_with_buffer_q.py 94; replay_buffer.buffer: {replay_buffer.buffer}")

            epi_return +=  rew
            agent.learn()
            obs = next_obs
            state = next_state

        print('Episode | {:5d} | Return | {:5.2f}'.format(i + 1, epi_return))
        epi_returns.append(epi_return)

    # Plot Smoothed Cumulative Reward (Return)
    plt.plot(get_running_mean(epi_returns))

plt.legend(REPLAY_BUFFER_SIZES)
plt.xlabel('Episode')
plt.ylabel('$G$')
os.makedirs('images',exist_ok=True)
if 'env' in vars(env).keys():
    env = env.env
plt.savefig(f'images/yash_{REPLAY_BUFFER_TYPE}_{env.spec._env_name}.png')
# plt.show()
    
###MODIFIED###
