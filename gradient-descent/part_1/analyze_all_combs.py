import numpy as np
from typing import Tuple


def increment_binary_mask(bin_mask: np.ndarray, n: int) -> None:
    for i in reversed(range(n)):
        if bin_mask[i] == 0:
            bin_mask[i] = 1
            break
        else:
            bin_mask[i] = 0


def analyze_all_combs(
    mass: np.ndarray, mass_limit: float, price: np.ndarray
) -> Tuple[int, int]:
    bin_length = len(mass)
    max_price = 0
    max_mass = 0
    binary_mask = np.zeros(bin_length, dtype=int)
    for _ in range(2**bin_length):
        temp_mass = 0
        temp_value = 0
        temp_mass = np.dot(binary_mask, mass)
        temp_value = np.dot(binary_mask, price)

        # print(bin_str, ':', temp_value)
        if temp_value > max_price and temp_mass <= mass_limit:
            max_price = temp_value
            max_mass = temp_mass
        increment_binary_mask(binary_mask, bin_length)
    return max_price, max_mass
