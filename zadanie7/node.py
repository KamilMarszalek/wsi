class Node:
    def __init__(self, vertex):
        self.vertex = vertex
        self.parents = []
        self.children = []
        self.cpt = {}