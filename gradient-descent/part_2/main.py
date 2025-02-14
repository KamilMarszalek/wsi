from steepest_ascent import steepest_ascent_classic, steepest_ascent_barzilai_borwein
from plot import draw_contour, description, draw_arrows
from cec2017.functions import f1, f2, f3
from booth import booth
import matplotlib.pyplot as plt
import numpy as np
from constants import LIMIT, DIMENSIONS, MAX_ITER, COLORS
from concurrent.futures import ThreadPoolExecutor


# BETA = 1e-9  # f1
# BETA = 1e-18  # f2
BETA = 1e-9  # f3
# BETA = 0.03  # booth
FUNCTION = f2
DIM_1 = 0
DIM_2 = 1


def worker(i: int) -> bool:
    x = np.random.uniform(-LIMIT, LIMIT, DIMENSIONS)
    print(f"Start {i}: ", x)
    points = steepest_ascent_barzilai_borwein(x, FUNCTION, BETA)
    print(f"End {i}")
    print(f"Function value {i}:", FUNCTION(points[-1]))
    print(f"End {i}: ", points[-1])
    draw_arrows(points, COLORS[i], DIM_1, DIM_2)
    description(x, points[-1], FUNCTION(points[-1]), BETA, i, COLORS[i], DIM_1, DIM_2)
    return len(points) < MAX_ITER


if __name__ == "__main__":
    found_min = False
    draw_contour(FUNCTION, LIMIT, 1, DIM_1, DIM_2)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(worker, range(3)))

    found_min = any(results)
    print("Found min:", found_min)
    plt.show()
