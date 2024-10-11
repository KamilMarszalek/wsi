import time
import numpy as np
from analyze_all_combs import analyze_all_combs
from heuristic_solution import heuristic_solution
from random import randint

NUM_OF_ELEMENTS = 5

m = np.array([randint(1, 100) for _ in range(NUM_OF_ELEMENTS)])  # mass of the objects
M = np.sum(m) / 2  # maximum mass of the objects
p = np.array([randint(1, 100) for _ in range(NUM_OF_ELEMENTS)])  # price of the objects


def main():
    start = time.process_time()
    # result = analyze_all_combs(m, M, p)
    result = heuristic_solution(m, M, p)
    end = time.process_time()
    total = end - start
    print("Max price:", result[0])
    print("Mass of prods:", result[1])
    print("Time:", "{0:02f}s".format(total))


if __name__ == "__main__":
    main()
