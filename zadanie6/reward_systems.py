def default_reward_system(
    reward: float, terminated: bool, truncated: bool, old_state: int, new_state: int
) -> float:
    return reward


def negative_hole_reward_system(
    reward: float, terminated: bool, truncated: bool, old_state: int, new_state: int
) -> float:
    if reward == 0 and (terminated or truncated):
        return -1
    return reward


def negative_hole_and_stagnation(
    reward: float, terminated: bool, truncated: bool, old_state: int, new_state: int
) -> float:
    if reward == 0 and ((terminated or truncated) or old_state == new_state):
        return -1
    return reward
