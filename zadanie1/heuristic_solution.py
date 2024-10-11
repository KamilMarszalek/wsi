from heapq import heappush, heappop
from node import Node
import numpy as np


def heuristic_solution(mass, mass_limit, price):
    max_price = 0
    temp_mass = 0

    dtype = [("mass", int), ("price", int), ("price_to_mass_ratio", float)]
    nodes = np.array(
        [(mass[i], price[i], price[i] / mass[i]) for i in range(len(mass))], dtype=dtype
    )
    # nodes.sort(reverse=True)
    nodes = np.sort(nodes, order="price_to_mass_ratio")[::-1]
    for node in nodes:
        if temp_mass + node["mass"] <= mass_limit:
            temp_mass += node["mass"]
            max_price += node["price"]
        if mass_limit - temp_mass < 1:
            break
    return max_price, temp_mass
    # max_price = 0
    # temp_mass = 0

    # nodes = [Node(mass[i], price[i], price[i] / mass[i]) for i in range(len(mass))]
    # nodes.sort(reverse=True)
    # for node in nodes:
    #     if temp_mass + node.mass <= mass_limit:
    #         temp_mass += node.mass
    #         max_price += node.price
    #     if mass_limit - temp_mass < 1:
    #         break
    # return max_price, temp_mass

    # nodes = []

    # for i in range(len(mass)):
    #     heappush(
    #         nodes, (-price[i] / mass[i], Node(mass[i], price[i], price[i] / mass[i]))
    #     )
    # while nodes and temp_mass < mass_limit:
    #     _, node = heappop(nodes)
    #     if temp_mass + node.mass <= mass_limit:
    #         temp_mass += node.mass
    #         max_price += node.price
    #     if mass_limit - temp_mass < 1:
    #         break

    # return max_price, temp_mass
