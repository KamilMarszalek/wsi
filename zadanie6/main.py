import gymnasium as gym
import numpy as np
from qlearning import qlearning
from reward_systems import *

env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

gamma = 0.95
beta = 0.1
epsilon_init = 1.0
epsilon_delta = 0.00001
epsilon_min = 0.01
epsilon_decay = 0.995

num_of_ind_runs = 25
num_episodes = 1000
averaged_reward = np.zeros(num_episodes)

print(env.unwrapped.desc)
averaged_reward = qlearning(env, num_episodes, beta, gamma, epsilon_init, epsilon_decay, epsilon_min, num_of_ind_runs, negative_hole_and_stagnation)
averaged_reward_base = qlearning(env, num_episodes, beta, gamma, epsilon_init, epsilon_decay, epsilon_min, num_of_ind_runs, default_reward_system)
print("Averaged rewards over episodes:", averaged_reward)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(averaged_reward_base, 'r')
plt.plot(averaged_reward, 'b')
plt.show()
