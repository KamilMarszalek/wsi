import matplotlib.pyplot as plt
import numpy as np
from constants import ARROW_SIZE, DIMENSIONS


def draw_contour(
    function, limit, plot_step, dimension_1=0, dimension_2=1, dimensions=DIMENSIONS
):
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


def draw_arrow(point_a, point_b, dimension_1, dimension_2, color):
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


def draw_arrows(points, color, dimension_1=0, dimension_2=1):
    for i in range(len(points) - 1):
        draw_arrow(points[i], points[i + 1], dimension_1, dimension_2, color)


def name_axes(dimension_1, dimension_2):
    plt.xlabel(f"X{dimension_1}")
    plt.ylabel(f"X{dimension_2}")


def description(
    starting_point,
    end_point,
    end_value,
    beta,
    counter,
    color,
    dimension_1=0,
    dimension_2=1,
):
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
