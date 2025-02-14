import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List
from constants import ARROW_SIZE, DIMENSIONS


def draw_contour(
    function: Callable[[np.ndarray], float],
    limit: float,
    plot_step: float,
    dimension_1: int = 0,
    dimension_2: int = 1,
    dimensions: int = DIMENSIONS,
) -> None:
    x_arr = np.arange(-limit, limit, plot_step)
    y_arr = np.arange(-limit, limit, plot_step)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.zeros(dimensions)
            point[dimension_1] = X[i, j]
            point[dimension_2] = Y[i, j]
            Z[i, j] = function(point)
    plt.contour(X, Y, Z, 20)
    name_axes(dimension_1, dimension_2)


def draw_arrow(
    point_a: np.ndarray,
    point_b: np.ndarray,
    dimension_1: int,
    dimension_2: int,
    color: str,
) -> None:
    dx = point_b[dimension_1] - point_a[dimension_1]
    dy = point_b[dimension_2] - point_a[dimension_2]
    head_width = ARROW_SIZE
    head_length = ARROW_SIZE
    plt.arrow(
        point_a[dimension_1],
        point_a[dimension_2],
        dx,
        dy,
        head_width=head_width,
        head_length=head_length,
        fc=color,
        ec=color,
    )


def draw_arrows(
    points: List[np.ndarray], color: str, dimension_1: int = 0, dimension_2: int = 1
) -> None:
    for i in range(len(points) - 1):
        draw_arrow(points[i], points[i + 1], dimension_1, dimension_2, color)


def name_axes(dimension_1: int, dimension_2: int) -> None:
    plt.xlabel(f"X{dimension_1}")
    plt.ylabel(f"X{dimension_2}")


def description(
    starting_point: np.ndarray,
    end_point: np.ndarray,
    end_value: float,
    beta: float,
    counter: int,
    color: str,
    dimension_1: int = 0,
    dimension_2: int = 1,
) -> None:
    description_text = (
        f"Start: [{starting_point[dimension_1]:.2f}, {starting_point[dimension_2]:.2f}]  "
        f"End: [{end_point[dimension_1]:.2f}, {end_point[dimension_2]:.2f}]  "
        f"End value: {end_value:.2f}  Beta: {beta}"
    )
    description_text = description_text.replace(".", ",")
    plt.text(
        0,
        123 - counter * 10,
        description_text,
        fontsize=10,
        color=color,
        ha="center",
    )
