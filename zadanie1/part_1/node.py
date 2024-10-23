class Node:
    def __init__(self, mass: float, price: float, ratio: float) -> None:
        self.mass = mass
        self.price = price
        self.ratio = ratio

    def __lt__(self, other: 'Node') -> bool:
        return self.ratio < other.ratio

    def __gt__(self, other: 'Node') -> bool:
        return self.ratio > other.ratio
