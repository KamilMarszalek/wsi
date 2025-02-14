import gymnasium as gym
import numpy as np
from qlearning import qlearning
from reward_systems import *
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

gamma = 0.95
beta = 0.2
epsilon_init = 1.0
epsilon_min = 0.01
epsilon_decay = 0.97

num_of_ind_runs = 25
num_episodes = 1000
averaged_reward = np.zeros(num_episodes)

averaged_reward = qlearning(
    env,
    num_episodes,
    beta,
    gamma,
    epsilon_init,
    epsilon_decay,
    epsilon_min,
    num_of_ind_runs,
    negative_hole_reward_system,
)
averaged_reward_base = qlearning(
    env,
    num_episodes,
    beta,
    gamma,
    epsilon_init,
    epsilon_decay,
    epsilon_min,
    num_of_ind_runs,
    default_reward_system,
)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines["left"].set_position("center")
ax.spines["bottom"].set_position("zero")
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
plt.plot(averaged_reward_base, "r")
plt.plot(averaged_reward, "b")
plt.show()
