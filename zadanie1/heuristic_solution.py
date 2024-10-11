from node import Node


def heuristic_solution(mass, mass_limit, price):
    max_price = 0
    temp_mass = 0
    nodes = [Node(mass[i], price[i], price[i] / mass[i]) for i in range(len(mass))]
    nodes.sort(reverse=True)
    for node in nodes:
        if temp_mass + node.mass <= mass_limit:
            temp_mass += node.mass
            max_price += node.price
        if mass_limit - temp_mass < 1:
            break
    return max_price, temp_mass
