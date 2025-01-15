import gymnasium as gym
import numpy as np

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

for run in range(num_of_ind_runs):
    qtable = np.zeros((state_size, action_size))
    epsilon = epsilon_init

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = 0
        done = False
        total_reward = 0

        for step in range(200):
            if np.random.uniform(0, 1) < epsilon or np.sum(qtable[state]) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state])

            new_state, reward, terminated, truncated, info = env.step(action)
            new_state = int(new_state)
            if (new_state) == 63:
                print("Reached target: ", episode)
                reward = 1
            delta = reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action]
            qtable[state, action] += beta * delta
            total_reward += reward
            state = new_state
            if terminated or truncated:
                break

        averaged_reward[episode] += total_reward
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

averaged_reward /= num_of_ind_runs
averaged_reward_base = np.copy(averaged_reward)
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
