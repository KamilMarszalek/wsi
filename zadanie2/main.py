from cec2017.functions import f2, f13
from evolutionary_algorithm import evolutionary_algorithm
import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt

FES = 50000
FUNCTION = f13
POPULATION_SIZE = 20
MUTATION_POWER = 1
POPULATION_SET = [2**n for n in range(10)]


def main():
    result_dict = {}
    for _ in range(200):
        best_point, best_value = evolutionary_algorithm(
            FUNCTION, POPULATION_SIZE, MUTATION_POWER, FES
        )
        result_dict[best_value] = best_point
        # print("Optimum: ", best_point)
        # print("Value: ", best_value)
    min_value = min(result_dict.keys())
    print("Min value: ", min_value)
    print(result_dict.get(min_value))


def generate_data():
    data_1 = {"population size": [], "max": [], "min": [], "mean": [], "std": []}
    data_2 = {"mutation power": [], "max": [], "min": [], "mean": [], "std": []}
    for i in POPULATION_SET:
        values = []
        for _ in range(25):
            _, value = evolutionary_algorithm(FUNCTION, i, MUTATION_POWER, FES)
            values.append(value)
        data_1["population size"].append(i)
        data_1["max"].append(max(values))
        data_1["min"].append(min(values))
        data_1["mean"].append(np.mean(values))
        data_1["std"].append(np.std(values))
    for i in range(1, 40):
        values = []
        for _ in range(25):
            _, value = evolutionary_algorithm(FUNCTION, POPULATION_SIZE, i / 10, FES)
            values.append(value)
        data_2["mutation power"].append(i / 10)
        data_2["max"].append(max(values))
        data_2["min"].append(min(values))
        data_2["mean"].append(np.mean(values))
        data_2["std"].append(np.std(values))
    df_1 = pd.DataFrame(data_1)
    name_1 = f"population_{FUNCTION.__name__}.csv"
    df_1.to_csv(name_1, index=False)
    df_2 = pd.DataFrame(data_2)
    name_2 = f"mutation_power_{FUNCTION.__name__}.csv"
    df_2.to_csv(name_2, index=False)


def format_dataframe(df: pd.DataFrame, decimal_places: int = 2) -> pd.DataFrame:
    return df.map(lambda x: f"{x:.{decimal_places}f}".replace(".", ","))


def create_table(filename: str, dataframe: pd.DataFrame) -> None:
    format_dataframe(dataframe)
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


if __name__ == "__main__":
    generate_data()
    df_1 = pd.read_csv(f"population_{FUNCTION.__name__}.csv")
    df_2 = pd.read_csv(f"mutation_power_{FUNCTION.__name__}.csv")
    create_table(f"population_{FUNCTION.__name__}", df_1)
    create_table(f"mutation_power_{FUNCTION.__name__}", df_2)
