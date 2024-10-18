from autograd import grad
import numpy as np
from constants import LIMIT, MAX_ITER, PRECISION


def steepest_ascent(
    point, function, beta, precision=PRECISION, minimize=True, limit=LIMIT
):
    points = [point]
    grad_fct = grad(function)
    step = np.subtract if minimize else np.add
    counter = 0
    while not stop(function, grad_fct(points[-1]), points, precision):
        gradients = grad_fct(points[-1])
        points.append(np.clip(step(points[-1], beta * gradients), -limit, limit))
        # print("Point:", points[-1])
        # print("len of vect: ", np.linalg.norm(gradients))
        counter += 1
        if (counter % 5000 == 0 and counter < 10000) or (
            counter % 1000 == 0 and counter >= 10000
        ):
            beta *= 2
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
