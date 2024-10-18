from steepest_ascent import steepest_ascent
from plot import draw_contour, description, draw_arrows
from cec2017.functions import f1, f2, f3
from booth import booth
import matplotlib.pyplot as plt
import numpy as np
from constants import LIMIT, DIMENSIONS, BETA, MAX_ITER, COLORS

FUNCTION = f1

if __name__ == "__main__":
    found_min = False
    draw_contour(FUNCTION, LIMIT, 1)
    for i in range(3):
        x = np.random.uniform(-LIMIT, LIMIT, DIMENSIONS)
        points = steepest_ascent(x, FUNCTION, BETA)
        print("End")
        print("Function value:", FUNCTION(points[-1]))
        print("End: ", points[-1])
        draw_arrows(points, COLORS[i])
        description(x, points[-1], FUNCTION(points[-1]), BETA, i, COLORS[i])
        if len(points) < MAX_ITER:
            found_min = True
    print("Found min:", found_min)
    plt.show()
