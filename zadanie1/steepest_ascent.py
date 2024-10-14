from autograd import grad
import numpy as np
import cec2017
from cec2017.functions import f1
from plot import plot, draw_arrow


def steepest_ascent(point, function, beta, precision=0.000001, minimize=True, limit=10):
    points = [point]
    grad_fct = grad(function)
    step = np.subtract if minimize > 0 else np.add
    while not stop(grad_fct(points[-1]), points, precision):
        gradients = grad_fct(points[-1])
        points.append(np.clip(-limit, limit, step(points[-1], beta * gradients)))
    return points


def stop(gradient, points, precision):
    if len(points) > 10000:
        return True
    return np.linalg.norm(gradient) < precision


if __name__ == "__main__":

    def booth(x):
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

    x = np.random.uniform(-10, 10, 2)
    points = steepest_ascent(x, booth, 0.08)
    print("End")
    # print("Points:", points)
    print("Function value:", f1(points[-1]))
    print("End: ", points[-1])
    plot(booth, 10, 0.1, points)
