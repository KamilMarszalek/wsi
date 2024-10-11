import time 
import numpy as np
from analyze_all_combs import analyze_all_combs

m = np.array([8, 3, 5, 2]) # mass of the objects
M = np.sum(m) / 2 # maximum mass of the objects
p = np.array([16, 8, 9, 6]) # price of the objects
def main():
    print("Hello world!")
    analyze_all_combs(m, M, p)


if __name__ == "__main__":
    main()