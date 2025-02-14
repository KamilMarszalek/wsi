import random
from collections import Counter, deque
import pandas as pd
import sys
import numpy as np
from node import Node
sys.path.append("/home/kamil/wsi")
import id3.main as id3

SAMPLES = 20000
CSV_FILE = "bayesian-network/assets/ache.csv"
NETWORK_FILE = "bayesian-network/assets/network.txt"


def load_network_from_file(file_name: str = NETWORK_FILE) -> dict[str, "Node"]:
    with open(file_name) as file_handle:
        nodes = file_handle.readline()[6:]
        vertices = {node.strip(): Node(node.strip()) for node in nodes.split(sep=",")}
        edges = file_handle.readline()[6:].strip()
        edges = [edge.strip().split(sep="->") for edge in edges.split(sep=",")]
        cpts = [line.strip() for line in file_handle.readlines()]
        set_children_and_parents(edges, vertices)
        set_cpt(vertices, cpts)
        return vertices


def set_children_and_parents(
    edges: list[list[str]], vertices: dict[str, "Node"]
) -> None:
    for edge in edges:
        parent_vertex = vertices[edge[0]]
        child_vertex = vertices[edge[1]]
        parent_vertex.children.append(child_vertex)
        child_vertex.parents.append(parent_vertex)


def set_cpt(vertices: dict[str, "Node"], cpts: list[str]) -> None:
    for cpt in cpts:
        expr, value = [rec.strip() for rec in cpt.split(" = ")]
        vertex_name = expr.split("=")[0][2:]
        expr = expr[2:-1]
        if vertex_name in vertices:
            vertices[vertex_name].cpt[expr] = float(value)


def topological_sort(vertices: dict[str, "Node"]) -> list[str]:
    in_degree = {node: 0 for node in vertices}
    for vertex in vertices.values():
        for child in vertex.children:
            in_degree[child.vertex] += 1

    queue = deque([v for v, deg in in_degree.items() if deg == 0])
    sorted_nodes = []

    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for child in vertices[node].children:
            in_degree[child.vertex] -= 1
            if in_degree[child.vertex] == 0:
                queue.append(child.vertex)

    return sorted_nodes


def generate_sample(vertices: dict[str, "Node"]) -> dict[str, str]:
    sorted_nodes = topological_sort(vertices)
    values = {}

    for node_name in sorted_nodes:
        node = vertices[node_name]
        if not node.parents:
            prob = node.cpt[f"{node_name}=true"]
        else:
            conditions = ",".join(
                [f"{parent.vertex}={values[parent.vertex]}" for parent in node.parents]
            )
            prob = node.cpt[f"{node_name}=true|{conditions}"]
        values[node_name] = "true" if random.random() < prob else "false"

    return values


def generate_sample_without_sorting(vertices: dict[str, "Node"]) -> dict[str, str]:
    values = {}
    unvisited = set(vertices.keys())

    while unvisited:
        for node_name in list(unvisited):
            node = vertices[node_name]
            if all(parent.vertex in values for parent in node.parents):
                if not node.parents:
                    prob = node.cpt[f"{node_name}=true"]
                else:
                    conditions = ",".join(
                        [
                            f"{parent.vertex}={values[parent.vertex]}"
                            for parent in node.parents
                        ]
                    )
                    prob = node.cpt[f"{node_name}=true|{conditions}"]
                values[node_name] = "true" if random.random() < prob else "false"
                unvisited.remove(node_name)
    return values


def test_sampling(network: dict[str, "Node"], samples: int = SAMPLES) -> None:
    counter = Counter()
    condition_counter = Counter()

    for _ in range(samples):
        sample = generate_sample_without_sorting(network)

        for node_name, value in sample.items():
            parents_str = ",".join(
                f"{parent.vertex}={sample[parent.vertex]}"
                for parent in network[node_name].parents
            )
            if parents_str:
                event_key = f"{node_name}={value}|{parents_str}"
            else:
                event_key = f"{node_name}={value}"
            if parents_str:
                cond_key = f"{node_name}|{parents_str}"
            else:
                cond_key = node_name

            counter[event_key] += 1
            condition_counter[cond_key] += 1
    print("Test Results:")
    for node_name, node in network.items():
        print(f"Node: {node_name}")
        for condition, expected_prob in node.cpt.items():
            if "|" in condition:
                _, right_side = condition.split("|")
                event_key = condition

                cond_key = f"{node_name}|{right_side}"

                observed_count = counter[event_key]
                cond_count = condition_counter[cond_key]
                if cond_count == 0:
                    observed_frequency = 0.0
                else:
                    observed_frequency = observed_count / cond_count

            else:
                event_key = condition
                observed_count = counter[event_key]
                observed_frequency = observed_count / samples

            print(
                f"  Condition: {condition}, Observed: {observed_frequency:.4f}, "
                f"Expected: {expected_prob:.4f}"
            )


def generate_data(
    network: dict[str, "Node"], samples: int = 10000, output_file: str = CSV_FILE
) -> None:
    data = [generate_sample(network) for _ in range(samples)]
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    network = load_network_from_file()
    test_sampling(network)
    generate_data(network)
    results = []
    matrices = []
    for i in range(100):
        accuracy, matrix = id3.test_model(CSV_FILE, 0.6, 3, 0, 3)
        results.append(accuracy)
        matrices.append(matrix)
    mean_accuracy = np.mean(results)
    min_accuracy = np.min(results)
    max_accuracy = np.max(results)
    std_accuracy = np.std(results)
    print("Mean accuracy:", mean_accuracy)
    print("Min accuracy:", min_accuracy)
    print("Max accuracy:", max_accuracy)
    print("Std accuracy:", std_accuracy)
    mean_matrix = np.mean(matrices, axis=0)
    print("Mean confusion matrix:")
    df_cm = pd.DataFrame(mean_matrix, index=["P", "N"], columns=["PP", "PN"])
    print(df_cm, "\n")