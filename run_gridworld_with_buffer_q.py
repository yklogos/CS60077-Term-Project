###MODIFIED###
#!/usr/bin/env python3
import copy
import gym
import matplotlib.pyplot as plt
import numpy as np
import time

from agents import TabularBufferQAgent
from replay import UniformReplayBuffer, CombinedReplayBuffer
from common import get_running_mean
from envs import get_grid_world


# init training variables
MAX_STEPS = None
REPLAY_BUFFER_TYPE = 'uniform'  # ['combined', 'uniform']
NB_EPISODES = 500
SEED = 0xc0ffee
REPLAY_BUFFER_SIZES = [100, 1000, 10000, 100000, 1000000][:1]

# init enviornment variables
env = get_grid_world(seed=SEED)
NUM_STATES = len(env.env.grid.grid)
NUM_ACTIONS = len(list(env.env.actions))
ENV_WIDTH = env.env.width
actions_idx_dict = {v:i for i,v in enumerate(list(env.env.actions))}

def pos_to_state(pos):  # pos: (row, col)
    return pos[0]*ENV_WIDTH + pos[1]

# init q_network value of terminal states to 0
start_q_network = np.random.rand(NUM_STATES, NUM_ACTIONS)
for i,cell in enumerate(env.env.grid.grid):  
    if cell is None:
        continue
    if cell.type == "goal":
        start_q_network[pos_to_state(cell.init_pos)] = np.zeros(NUM_ACTIONS)

# For reproducibility
np.random.seed(SEED)


replay_buffer_dict = {'uniform': UniformReplayBuffer, 'combined': CombinedReplayBuffer}

for replay_buffer_size in REPLAY_BUFFER_SIZES:

    # Setup Environment
    env = get_grid_world(seed=SEED)

    # Setup Agent
    q_network = copy.deepcopy(start_q_network)
    
    buffer = replay_buffer_dict[REPLAY_BUFFER_TYPE]
    replay_buffer = buffer(replay_buffer_size)
    agent = TabularBufferQAgent(q_network, replay_buffer, range(NUM_STATES), env.action_space, alpha=0.1, epsilon=0.1, gamma=1, batch_size=10)

    # Train agent
    epi_returns = []
    for i in range(NB_EPISODES):
        obs = env.reset()
        info = {'timeout': False}
        state = pos_to_state(env.agent_pos)
        epi_return = 0
        
        done = False

        while not done:
            
            if MAX_STEPS and not info['timeout']:
                break
            
            action = agent.get_action(state, is_table=True)
            next_obs, rew, done, info = env.step(action)
            next_state = pos_to_state(env.agent_pos)
            transition = (state, action, next_state, rew, done)
            replay_buffer.append(transition)
            epi_return +=  rew
            agent.learn()
            obs = next_obs
        
        print('Episode | {:5d} | Return | {:5.2f}'.format(i + 1, epi_return))
        epi_returns.append(epi_return)

    # Plot Smoothed Cumulative Reward (Return)
    plt.plot(get_running_mean(epi_returns))

plt.legend(REPLAY_BUFFER_SIZES)
plt.xlabel('Episode')
plt.ylabel('$G$')
plt.savefig('images/run_gridworld_with_buffer_q_smooth.png')
plt.show()
###MODIFIED###
