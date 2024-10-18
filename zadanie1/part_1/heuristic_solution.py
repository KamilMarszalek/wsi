from heapq import heappush, heappop
from node import Node


def heuristic_solution(mass, mass_limit, price):
    max_price = 0
    temp_mass = 0
    nodes = []
    for i in range(len(mass)):
        heappush(
            nodes, (-price[i] / mass[i], Node(mass[i], price[i], price[i] / mass[i]))
        )
    while nodes and temp_mass < mass_limit:
        _, node = heappop(nodes)
        if temp_mass + node.mass <= mass_limit:
            temp_mass += node.mass
            max_price += node.price
        if mass_limit - temp_mass < 1:
            break

    return max_price, temp_mass
