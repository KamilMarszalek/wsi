class Node:
    def __init__(self, mass, price, ratio):
        self.mass = mass
        self.price = price
        self.ratio = ratio

    def __lt__(self, other):
        return self.ratio < other.ratio

    def __gt__(self, other):
        return self.ratio > other.ratio
