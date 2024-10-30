import numpy as np
from typing import Callable, Tuple

LIMIT = 100
DIMENSIONS = 10


def evolutionary_algorithm(
    function: Callable[[np.ndarray], float],
    population_size: int,
    mutation_power: float,
    fes: int,
) -> Tuple[np.ndarray, float]:
    counter = 1
    population = generate_population(population_size)
    max_iter = fes // population_size
    evaluation = get_eval(population, function)
    best_point, best_value = find_best(population, evaluation)
    while not stop(counter, max_iter):
        reproduction_result = reproduction(population, evaluation, population_size)
        mutation_result = mutation(reproduction_result, mutation_power)
        mutation_evaluation = get_eval(mutation_result, function)
        best_point_m, value_m = find_best(mutation_result, mutation_evaluation)
        if value_m < best_value:
            best_value = value_m
            best_point = best_point_m
        population = mutation_result
        evaluation = mutation_evaluation
        counter += 1
    return best_point, best_value


def generate_population(population_size: int) -> np.ndarray:
    return np.random.uniform(-LIMIT, LIMIT, (population_size, DIMENSIONS))


def reproduction(
    population: np.ndarray, evaluation: np.ndarray, population_size: int
) -> np.ndarray:
    rng = np.random.default_rng()
    indices = rng.integers(0, population_size, size=(population_size, 2))
    selected = np.where(
        evaluation[indices[:, 0]] < evaluation[indices[:, 1]],
        indices[:, 0],
        indices[:, 1],
    )
    return population[selected]


def mutation(reproduction: np.ndarray, mutation_power: float) -> np.ndarray:
    mutation = reproduction + mutation_power * np.random.uniform(
        -1, 1, reproduction.shape
    )
    return np.clip(mutation, -LIMIT, LIMIT)


def stop(counter: int, max_iter: int) -> bool:
    return counter >= max_iter


def get_eval(
    population: np.ndarray, function: Callable[[np.ndarray], float]
) -> np.ndarray:
    return np.apply_along_axis(function, 1, population)


def find_best(
    population: np.ndarray, evaluation: np.ndarray
) -> Tuple[np.ndarray, float]:
    best_index = np.argmin(evaluation)
    return population[best_index], evaluation[best_index]
