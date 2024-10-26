from cec2017.functions import f2, f13
from evolutionary_algorithm import evolutionary_algorithm

FES = 50000
FUNCTION = f13
POPULATION_SIZE = 20
MUTATION_POWER = 1.2


def main():
    result_dict = {}
    for _ in range(100):
        best_point, best_value = evolutionary_algorithm(
            FUNCTION, POPULATION_SIZE, MUTATION_POWER, FES
        )
        result_dict[best_value] = best_point
        # print("Optimum: ", best_point)
        # print("Value: ", best_value)
    min_value = min(result_dict.keys())
    print("Min value: ", min_value)
    print(result_dict.get(min_value))


if __name__ == "__main__":
    main()
