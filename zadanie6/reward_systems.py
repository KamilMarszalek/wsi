def default_reward_system(reward, terminated, truncated, old_state, new_state):
    return reward

def negative_hole_reward_system(reward, terminated, truncated, old_state, new_state):
    if reward == 0 and (terminated or truncated):
        return -1
    return reward

def negative_hole_and_stagnation(reward, terminated, truncated, old_state, new_state):
    if reward == 0 and (terminated or truncated) or old_state == new_state:
        return -1
    return reward