###MODIFIED###
#!/usr/bin/env python3

# TODO:
#     1. verify results
#     2. find a way to remove unecessary actions
#     3. implement PEB

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

# init training variables
ALPHA = 0.1
BATCH_SIZE = 10
EPSILON = 0.1
ENV_NAMES = ['MiniGrid-Empty-16x16-v0', 'MiniGrid-FourRooms-v0']
GAMMA = 1
MAX_STEPS = None
NB_EPISODES = 100
REPLAY_BUFFER_SIZES = [100, 1000, 10000, 100000, 1000000]
REPLAY_BUFFER_TYPE = 'combined'  # ['combined', 'uniform']
REPLAY_BUFFER_TYPES = ['combined', 'uniform']

replay_buffer_dict = {'combined': CombinedReplayBuffer, 'uniform': UniformReplayBuffer}
for env_name in ENV_NAMES:
    
    # init enviornment variables
    env = get_grid_world(env_name=env_name, seed=SEED)
    if 'env' in vars(env).keys():
        env = env.env
    NUM_STATES = len(env.grid.grid)
    ACTIONS = list(env.actions)
    NUM_ACTIONS = len(ACTIONS)
    ENV_WIDTH = env.width
    start_q_network = np.zeros((NUM_STATES, NUM_ACTIONS)) # init all values to 0
    # # init q_network value of terminal states to 0
    # for i,cell in enumerate(env.env.grid.grid):  
    #     if cell is None:
    #         continue
    #     if cell.type == "goal":
    #         start_q_network[pos_to_state(cell.init_pos)] = np.zeros(NUM_ACTIONS)
    
    def pos_to_state(pos):  # pos: (row, col)
        return pos[0]*ENV_WIDTH + pos[1]
    
    for buffer_type in ['combined', 'uniform']:
        
        for replay_buffer_size in REPLAY_BUFFER_SIZES:
            env = get_grid_world(max_steps=MAX_STEPS, seed=SEED)
            q_network = copy.deepcopy(start_q_network)
            replay_buffer = replay_buffer_dict[buffer_type](replay_buffer_size)
            agent = TabularBufferQAgent(q_network, replay_buffer, range(NUM_STATES), env.action_space, alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA, batch_size=BATCH_SIZE)
            epi_returns = []

            for i in range(NB_EPISODES):
                obs = env.reset()
                state = pos_to_state(env.agent_pos)
                epi_return = 0
                done = False

                while not done:
                    if MAX_STEPS:
                        if not info['timeout']:
                            break
                    action = agent.get_action(state, is_table=True)
                    next_obs, rew, done, info = env.step(action)
                    rew -= 1   # -1 reward at each step
                    next_state = pos_to_state(env.agent_pos)
                    transition = (state, action, next_state, rew, done) 
                    replay_buffer.append(transition)
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
        plt.savefig(f'images/yash_{buffer_type}_{env.spec._env_name}.png')
        # plt.show()

###MODIFIED###
