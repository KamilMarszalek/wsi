import numpy as np

LIMIT = 100
DIMENSIONS = 10


def evolutionary_algorithm(function, population_size, mutation_power, fes):
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


def generate_population(population_size):
    return [
        np.random.uniform(-LIMIT, LIMIT, DIMENSIONS) for _ in range(population_size)
    ]


def reproduction(population, evaluation, population_size):
    reproduction = []
    rng = np.random.default_rng()
    for _ in range(population_size):
        index_1 = rng.integers(0, population_size)
        index_2 = rng.integers(0, population_size)
        point_1 = population[index_1]
        point_2 = population[index_2]
        reproduction.append(
            point_1 if evaluation[index_1] < evaluation[index_2] else point_2
        )
    return reproduction


def mutation(reproduction, mutation_power):
    mutation = []
    for point in reproduction:
        temp = np.random.uniform(-1, 1, DIMENSIONS)
        point = point - mutation_power * temp
        point = np.clip(point, -LIMIT, LIMIT)
        mutation.append(point)
    return mutation


def stop(counter, max_iter):
    return counter >= max_iter


def get_eval(population, function):
    return [function(point) for point in population]


def find_best(population, evaluation):
    best_point = population[0]
    best_eval = evaluation[0]
    for point, score in zip(population, evaluation):
        if score < best_eval:
            best_eval = score
            best_point = point
    return best_point, best_eval
