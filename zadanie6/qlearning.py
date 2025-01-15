import numpy as np

def qlearning(env, num_episodes, beta, gamma, epsilon_init, epsilon_decay, epsilon_min, num_of_ind_runs):
    state_size = env.observation_space.n
    action_size = env.action_space.n

    averaged_reward = np.zeros(num_episodes)

    for _ in range(num_of_ind_runs):
        qtable = np.zeros((state_size, action_size))
        epsilon = epsilon_init

        for episode in range(num_episodes):
            state, _ = env.reset()
            state = 0
            total_reward = 0

            for _ in range(200):
                if np.random.uniform(0, 1) < epsilon or np.sum(qtable[state]) == 0:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(qtable[state])

                new_state, reward, terminated, truncated, _ = env.step(action)
                delta = reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action]
                qtable[state, action] += beta * delta
                total_reward += reward
                state = new_state
                if terminated or truncated:
                    break

            averaged_reward[episode] += total_reward
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    averaged_reward /= num_of_ind_runs
    return averaged_reward