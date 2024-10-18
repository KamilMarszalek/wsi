from autograd import grad
import numpy as np
from cec2017.functions import f1, f2, f3
from booth import booth
from plot import plot


LIMIT = 100
MAX_ITER = 20000
FUNCTION = f3
GRAD_LIMIT = 100
PRECISION = 0.001
DIMENSIONS = 10
BETA = 0.00000002


def steepest_ascent(
    point, function, beta, precision=PRECISION, minimize=True, limit=LIMIT
):
    points = [point]
    grad_fct = grad(function)
    step = np.subtract if minimize else np.add
    while not stop(function, grad_fct(points[-1]), points, precision):
        # gradients = np.clip(grad_fct(points[-1]), -GRAD_LIMIT, GRAD_LIMIT)
        # gradients = grad_fct(points[-1]) / np.linalg.norm(grad_fct(points[-1]))
        gradients = grad_fct(points[-1])
        points.append(np.clip(step(points[-1], beta * gradients), -limit, limit))
        print("Point:", points[-1])
        print("len of vect: ", np.linalg.norm(gradients))
    return points


def stop(function, gradient, points, precision, precision2=PRECISION):
    if len(points) > MAX_ITER:
        return True
    if len(points) < 2:
        return False
    return (
        np.linalg.norm(gradient) < precision
        or np.linalg.norm(function(points[-1]) - function(points[-2])) < precision2
    )


if __name__ == "__main__":
    x = np.random.uniform(-LIMIT, LIMIT, DIMENSIONS)
    # x = np.array([0.0, 0.0])
    points = steepest_ascent(x, FUNCTION, BETA)
    print("End")
    print("Function value:", FUNCTION(points[-1]))
    print("End: ", points[-1])
    plot(FUNCTION, LIMIT, 1, points)