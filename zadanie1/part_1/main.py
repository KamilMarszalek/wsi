import time
import numpy as np
from analyze_all_combs import analyze_all_combs
from heuristic_solution import heuristic_solution
from random import randint
from enum import Enum


class Solution(Enum):
    ALL_COMBINATIONS = 0
    HEURISTIC = 1
    BOTH = 2


SOLUTION = Solution.BOTH

NUM_OF_ELEMENTS = 25


m = np.array([randint(1, 100) for _ in range(NUM_OF_ELEMENTS)])  # mass of the objects
M = np.sum(m) / 2  # maximum mass of the objects
p = np.array([randint(1, 100) for _ in range(NUM_OF_ELEMENTS)])  # price of the objects


def handle_generating_all_combinations():
    print("Generating all combinations:")
    start = time.process_time()
    result = analyze_all_combs(m, M, p)
    end = time.process_time()
    total = end - start
    print("Max price:", result[0], "Mass of prods:", result[1])
    print("Time:", "{0:02f}s".format(total), "\n")


def handle_heuristic_solution():
    print("Heuristic solution:")
    start = time.process_time()
    result = heuristic_solution(m, M, p)
    end = time.process_time()
    total = end - start
    print("Max price:", result[0], "Mass of prods:", result[1])
    print("Time:", "{0:02f}s".format(total))


def main():
    print("Mass limit:", M, "\n")

    if SOLUTION == Solution.ALL_COMBINATIONS:
        handle_generating_all_combinations()
    elif SOLUTION == Solution.HEURISTIC:
        handle_heuristic_solution()
    else:
        handle_generating_all_combinations()
        handle_heuristic_solution()


if __name__ == "__main__":
    main()
