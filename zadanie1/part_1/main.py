import time
import numpy as np
from analyze_all_combs import analyze_all_combs
from heuristic_solution import heuristic_solution
from random import randint
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt


class Solution(Enum):
    ALL_COMBINATIONS = 0
    HEURISTIC = 1
    BOTH = 2


SOLUTION = Solution.BOTH

NUM_OF_ELEMENTS = 20
ELEMENTS = [5, 10, 15, 20, 25]


m = np.array([randint(1, 100) for _ in range(NUM_OF_ELEMENTS)])  # mass of the objects
M = np.sum(m) / 2  # maximum mass of the objects
p = np.array([randint(1, 100) for _ in range(NUM_OF_ELEMENTS)])  # price of the objects


def handle_generating_all_combinations():
    start = time.process_time()
    result = analyze_all_combs(m, M, p)
    end = time.process_time()
    total = end - start
    return result[0], result[1], total


def handle_heuristic_solution():
    start = time.process_time()
    result = heuristic_solution(m, M, p)
    end = time.process_time()
    total = end - start
    return result[0], result[1], total


def print_results(result):
    print("Max price:", result[0], "Mass of prods:", result[1])
    print("Time:", "{0:02f}s".format(result[2]), "\n")


def main():
    print("Mass limit:", M, "\n")

    if SOLUTION == Solution.ALL_COMBINATIONS:
        print("Generating all combinations:")
        result = handle_generating_all_combinations()
        print_results(result)

    elif SOLUTION == Solution.HEURISTIC:
        print("Heuristic solution:")
        result = handle_heuristic_solution()
        print_results(result)
    else:
        print("Generating all combinations:")
        result = handle_generating_all_combinations()
        print_results(result)
        print("Heuristic solution:")
        result = handle_heuristic_solution()
        print_results(result)


def generate_data():
    data_1 = {"n": [], "max_price": [], "mass_of_prods": [], "time": []}
    data_2 = {"n": [], "max_price": [], "mass_of_prods": [], "time": []}

    for i in ELEMENTS:
        for _ in range(25):
            m = np.array([randint(1, 100) for _ in range(i)])  # mass of the objects
            M = np.sum(m) / 2  # maximum mass of the objects
            p = np.array([randint(1, 100) for _ in range(i)])  # price of the objects

            start = time.process_time()
            result = analyze_all_combs(m, M, p)
            end = time.process_time()
            total = end - start

            data_1["n"].append(i)
            data_1["max_price"].append(result[0])
            data_1["mass_of_prods"].append(result[1])
            data_1["time"].append(format(total, ".6f"))

            start = time.process_time()
            result = heuristic_solution(m, M, p)
            end = time.process_time()
            total = end - start

            data_2["n"].append(i)
            data_2["max_price"].append(result[0])
            data_2["mass_of_prods"].append(result[1])
            data_2["time"].append(format(total, ".6f"))

    df_1 = pd.DataFrame(data_1)
    df_1.to_csv("all_combinations.csv", index=False)
    df_2 = pd.DataFrame(data_2)
    df_2.to_csv("heuristic_solution.csv", index=False)


def create_table(filename, dataframe):
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.axis("tight")
    ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc="center",
        loc="center",
    )
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.clf()


def compare_max_price():
    all_combs_df = pd.read_csv("all_combinations.csv")
    heuristic_df = pd.read_csv("heuristic_solution.csv")
    max_price_all_combs = all_combs_df["max_price"]
    max_price_heuristics = heuristic_df["max_price"]
    comparison_df = pd.DataFrame(
        {
            "n": all_combs_df["n"],
            "PRICE ALL COMBINATIONS": max_price_all_combs,
            "PRICE HEURISTICS": max_price_heuristics,
            "DIFFERENCE": max_price_all_combs - max_price_heuristics,
        }
    )
    max_diff = []
    min_diff = []
    mean_diff = []
    for i in ELEMENTS:
        max_diff.append(max(comparison_df[comparison_df["n"] == i]["DIFFERENCE"]))
        min_diff.append(min(comparison_df[comparison_df["n"] == i]["DIFFERENCE"]))
        mean_diff.append(np.mean(comparison_df[comparison_df["n"] == i]["DIFFERENCE"]))
    table_df = pd.DataFrame(
        {
            "n": ELEMENTS,
            "MAX DIFFERENCE": max_diff,
            "MIN DIFFERENCE": min_diff,
            "MEAN DIFFERENCE": mean_diff,
        }
    )
    table_df = table_df.map(lambda x: f"{x:.2f}".replace(".", ","))
    create_table("comparison_table", table_df)


def table_of_times():
    all_combs_df = pd.read_csv("all_combinations.csv")
    heuristic_df = pd.read_csv("heuristic_solution.csv")
    times_all_combs = all_combs_df["time"]
    times_heuristics = heuristic_df["time"]
    max_time_all_combs = []
    min_time_all_combs = []
    mean_time_all_combs = []
    std_time_all_combs = []
    for i in ELEMENTS:
        max_time_all_combs.append(max(all_combs_df[all_combs_df["n"] == i]["time"]))
        min_time_all_combs.append(min(all_combs_df[all_combs_df["n"] == i]["time"]))
        mean_time_all_combs.append(
            np.mean(all_combs_df[all_combs_df["n"] == i]["time"])
        )
        std_time_all_combs.append(np.std(all_combs_df[all_combs_df["n"] == i]["time"]))
    temp = pd.DataFrame(
        {
            "n": ELEMENTS,
            "MINIMUM TIME": min_time_all_combs,
            "MAXIMUM TIME": max_time_all_combs,
            "MEAN TIME": mean_time_all_combs,
            "STANDARD DEVIATION": std_time_all_combs,
        }
    )
    temp = temp.map(lambda x: f"{x:.6f}".replace(".", ","))
    create_table("times_table_all_combinations", temp)
    max_time_heuristics = []
    min_time_heuristics = []
    mean_time_heuristics = []
    std_time_heuristics = []
    for i in ELEMENTS:
        max_time_heuristics.append(max(heuristic_df[heuristic_df["n"] == i]["time"]))
        min_time_heuristics.append(min(heuristic_df[heuristic_df["n"] == i]["time"]))
        mean_time_heuristics.append(
            np.mean(heuristic_df[heuristic_df["n"] == i]["time"])
        )
        std_time_heuristics.append(np.std(heuristic_df[heuristic_df["n"] == i]["time"]))
    temp = pd.DataFrame(
        {
            "n": ELEMENTS,
            "MINIMUM TIME": min_time_heuristics,
            "MAXIMUM TIME": max_time_heuristics,
            "MEAN TIME": mean_time_heuristics,
            "STANDARD DEVIATION": std_time_heuristics,
        }
    )
    temp = temp.map(lambda x: f"{x:.6f}".replace(".", ","))
    create_table("times_table_heuristics", temp)


if __name__ == "__main__":
    main()
    # generate_data()
    # df_1 = pd.read_csv("all_combinations.csv")
    # df_2 = pd.read_csv("heuristic_solution.csv")
    # compare = pd.concat([df_1, df_2], axis=1)
    # create_table("all_combinations", df_1)
    # create_table("heuristic_solution", df_2)
    # compare_max_price()
    # table_of_times()
