from cec2017.functions import f2, f13
from evolutionary_algorithm import evolutionary_algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple


FES = 50000
FUNCTION = f13
POPULATION_SIZE = 4
MUTATION_POWER = 1
POPULATION_SET = [2**n for n in range(10)]


def main() -> None:
    generate_data()
    df_1 = pd.read_csv(f"population_{FUNCTION.__name__}.csv")
    df_2 = pd.read_csv(f"mutation_power_{FUNCTION.__name__}.csv")
    create_table(f"population_{FUNCTION.__name__}", df_1)
    create_table(f"mutation_power_{FUNCTION.__name__}", df_2)


def run_evolutionary_algorithm(args: Tuple[int, float, int]) -> float:
    population_size, mutation_power, fes = args
    _, value = evolutionary_algorithm(FUNCTION, population_size, mutation_power, fes)
    return value


def generate_data() -> None:
    data_1 = {
        "rozmiar populacji - μ": [],
        "max": [],
        "min": [],
        "średnia": [],
        "std": [],
    }
    data_2 = {"siła mutacji - σ": [], "max": [], "min": [], "średnia": [], "std": []}
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i in POPULATION_SET:
            args = [(i, MUTATION_POWER, FES) for _ in range(100)]
            values = list(executor.map(run_evolutionary_algorithm, args))
            data_1["rozmiar populacji - μ"].append(i)
            data_1["max"].append(max(values))
            data_1["min"].append(min(values))
            data_1["średnia"].append(np.mean(values))
            data_1["std"].append(np.std(values))
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i in range(31, 41):
            mutation_power = i / 10
            args = [(POPULATION_SIZE, mutation_power, FES) for _ in range(100)]
            values = list(executor.map(run_evolutionary_algorithm, args))
            data_2["siła mutacji - σ"].append(mutation_power)
            data_2["max"].append(max(values))
            data_2["min"].append(min(values))
            data_2["średnia"].append(np.mean(values))
            data_2["std"].append(np.std(values))

    df_1 = pd.DataFrame(data_1)
    df_1["rozmiar populacji - μ"] = df_1["rozmiar populacji - μ"].astype(int)
    name_1 = f"population_{FUNCTION.__name__}.csv"
    df_1.to_csv(name_1, index=False)

    df_2 = pd.DataFrame(data_2)
    df_2["siła mutacji - σ"] = df_2["siła mutacji - σ"].map(
        lambda x: f"{x:.1f}".replace(".", ",")
    )
    name_2 = f"mutation_power_{FUNCTION.__name__}.csv"
    df_2.to_csv(name_2, index=False)


def format_dataframe(df: pd.DataFrame, decimal_places: int = 2) -> pd.DataFrame:
    df["max"] = df["max"].map(lambda x: f"{x:.{decimal_places}f}".replace(".", ","))
    df["min"] = df["min"].map(lambda x: f"{x:.{decimal_places}f}".replace(".", ","))
    df["średnia"] = df["średnia"].map(
        lambda x: f"{x:.{decimal_places}f}".replace(".", ",")
    )
    df["std"] = df["std"].map(lambda x: f"{x:.{decimal_places}f}".replace(".", ","))
    return df


def create_table(filename: str, dataframe: pd.DataFrame) -> None:
    dataframe = format_dataframe(dataframe)
    _, ax = plt.subplots()
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
    main()
