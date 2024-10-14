import matplotlib.pyplot as plt
import cec2017
import numpy as np

from cec2017.functions import f1


def plot(function, limit, plot_step, points):
    x_arr = np.arange(-limit, limit, plot_step)
    y_arr = np.arange(-limit, limit, plot_step)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = function([X[i, j], Y[i, j]])
    contour = plt.contour(X, Y, Z, 20)
    plt.colorbar()
    plt.clabel(contour, inline=True, fontsize=8)
    for i in range(len(points) - 1):
        draw_arrow(points[i], points[i + 1])
    plt.show()


def draw_arrow(point_a, point_b):
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    length = np.sqrt(dx**2 + dy**2)
    head_width = length * 0.05
    head_length = length * 0.05
    plt.arrow(
        point_a[0],
        point_a[1],
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
