import matplotlib.pyplot as plt
import cec2017
import numpy as np

from cec2017.functions import f1

ARROW_SIZE = 1


def plot(
    function, limit, plot_step, points, dimension_1=0, dimension_2=1, dimensions=10
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
    plt.colorbar()
    for i in range(len(points) - 1):
        draw_arrow(points[i], points[i + 1], dimension_1, dimension_2)
    plt.show()


def draw_arrow(point_a, point_b, dimension_1, dimension_2):
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
        fc="k",
        ec="k",
    )


if __name__ == "__main__":
    plot(f1, 10, 0.1)
    plt.show()
