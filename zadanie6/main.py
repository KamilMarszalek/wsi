import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

gamma = 0.9
beta = 0.9
epsilon_init = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01

num_of_ind_runs = 25
num_episodes = 1000
averaged_rewards = np.zeros(num_episodes)

for run in range(num_of_ind_runs):
    qtable = np.zeros((state_size, action_size))
    epsilon = epsilon_init

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = int(state)
        done = False
        total_reward = 0

        for step in range(200):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, done, _, _ = env.step(action)
            new_state = int(new_state)
            qtable[state, action] += beta * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            total_reward += reward
            state = new_state
            if done:
                break
        
        averaged_rewards[episode] += total_reward
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
averaged_rewards /= num_of_ind_runs
print("Averaged rewards over episodes:", averaged_rewards)
