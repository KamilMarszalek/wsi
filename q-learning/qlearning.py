import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Tuple
import gymnasium as gym


def choose_action(env: gym.Env, qtable: np.ndarray, state: int, epsilon: float) -> int:
    if np.random.uniform(0, 1) < epsilon or np.sum(qtable[state]) == 0:
        return env.action_space.sample()
    return np.argmax(qtable[state])


def single_run(
    env: gym.Env,
    num_episodes: int,
    beta: float,
    gamma: float,
    epsilon_init: float,
    epsilon_decay: float,
    epsilon_min: float,
    reward_system: Callable[[float, bool, bool, int, int], float],
) -> np.ndarray:
    state_size = env.observation_space.n
    action_size = env.action_space.n

    qtable = np.zeros((state_size, action_size))
    epsilon = epsilon_init

    averaged_reward = np.zeros(num_episodes)

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(200):
            action = choose_action(env, qtable, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            custom_reward = reward_system(
                reward, terminated, truncated, state, new_state
            )
            delta = (
                custom_reward
                + gamma * np.max(qtable[new_state, :])
                - qtable[state, action]
            )
            qtable[state, action] += beta * delta
            total_reward += reward
            state = new_state
            if terminated or truncated:
                break

        averaged_reward[episode] = total_reward
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return averaged_reward


def qlearning(
    env: gym.Env,
    num_episodes: int,
    beta: float,
    gamma: float,
    epsilon_init: float,
    epsilon_decay: float,
    epsilon_min: float,
    num_of_ind_runs: int,
    reward_system: Callable[[float, bool, bool, int, int], float],
) -> np.ndarray:
    with ProcessPoolExecutor() as executor:
        results = [
            executor.submit(
                single_run,
                env,
                num_episodes,
                beta,
                gamma,
                epsilon_init,
                epsilon_decay,
                epsilon_min,
                reward_system,
            )
            for _ in range(num_of_ind_runs)
        ]
        results = [result.result() for result in results]
    return np.mean(results, axis=0)
