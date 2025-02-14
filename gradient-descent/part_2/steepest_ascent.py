from autograd import grad
import numpy as np
from typing import Callable, List
from constants import LIMIT, MAX_ITER, PRECISION


def steepest_ascent_classic(
    point: np.ndarray,
    function: Callable[[np.ndarray], float],
    beta: float,
    precision: float = PRECISION,
    minimize: bool = True,
    limit: float = LIMIT,
) -> List[np.ndarray]:
    points = [point]
    grad_fct = grad(function)
    step = np.subtract if minimize else np.add
    while not stop(function, grad_fct(points[-1]), points, precision):
        gradients = grad_fct(points[-1])
        points.append(np.clip(step(points[-1], beta * gradients), -limit, limit))
    return points


def steepest_ascent_barzilai_borwein(
    point: np.ndarray,
    function: Callable[[np.ndarray], float],
    beta: float,
    precision: float = PRECISION,
    minimize: bool = True,
    limit: float = LIMIT,
) -> List[np.ndarray]:
    points = [point]
    grad_fct = grad(function)
    step = np.subtract if minimize else np.add
    gradients_1 = grad_fct(points[-1])
    points.append(np.clip(step(points[-1], beta * gradients_1), -limit, limit))
    gradients_2 = grad_fct(points[-1])
    while not stop(function, gradients_2, points, precision):
        beta = np.dot(points[-1] - points[-2], gradients_2 - gradients_1) / np.dot(
            gradients_2 - gradients_1, gradients_2 - gradients_1
        )
        gradients_1 = gradients_2
        points.append(np.clip(step(points[-1], beta * gradients_1), -limit, limit))
        gradients_2 = grad_fct(points[-1])
    return points


def stop(
    function: Callable[[np.ndarray], float],
    gradient: np.ndarray,
    points: List[np.ndarray],
    precision: float,
    precision2: float = PRECISION,
) -> bool:
    if len(points) > MAX_ITER:
        return True
    if len(points) < 2:
        return False
    return (
        np.linalg.norm(gradient) < precision
        or np.linalg.norm(function(points[-1]) - function(points[-2])) < precision2
    )
