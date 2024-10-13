from autograd import grad
import numpy as np
import cec2017
from cec2017.functions import f1


def steepest_ascent(point, function, beta, precision=0.000001):
    points = [point]
    while not stop(function, points, beta, precision):
        grad_fct = grad(function)
        gradients = grad_fct(points[-1])
        points.append(np.add(points[-1], beta * gradients))
    return points


def stop(function, points, beta, precision):
    if len(points) > 10000:
        return True
    if len(points) < 2:
        return False
    return abs(function(points[-1]) - function(points[-2])) < precision


if __name__ == "__main__":

    def booth(x):
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

    points = steepest_ascent(np.array([10, 10], dtype=float), booth, -0.0999)
    print("End")
    # print("Points:", points)
    print("Function value:", booth(points[-1]))
    print("End: ", points[-1])
