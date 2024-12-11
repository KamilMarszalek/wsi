import numpy as np
from collections import Counter


class Node:
    def __init__(self, attribute=None, target=None, children=None, default_label=None):
        self.attribute = attribute
        self.target = target
        self.children = children if children is not None else {}
        self.default_label = default_label


def entropy(targets):
    counts = Counter(targets).values()
    probabilities = [c / len(targets) for c in counts]
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


def most_common_label(targets):
    targets = [t for t in targets if t is not None]
    if not targets:
        return None
    return Counter(targets).most_common(1)[0][0]


def build(data, targets, attributes):
    if len(set(targets)) == 1:
        return Node(target=targets[0])
    if len(attributes) == 0:
        return Node(target=most_common_label(targets))

    current_entropy = entropy(targets)
    best_gain = 0
    best_feature = None
    best_splits = None

    for attribute in attributes:
        values = np.unique(data[:, attribute])
        splits = {}
        for value in values:
            subset_indices = data[:, attribute] == value
            splits[value] = targets[subset_indices]

        weighted_entropy = sum(
            (len(subset) / len(targets)) * entropy(subset) for subset in splits.values()
        )
        gain = current_entropy - weighted_entropy

        if gain > best_gain:
            best_gain = gain
            best_feature = attribute
            best_splits = splits

    if best_gain == 0:
        return Node(target=most_common_label(targets))

    children = {}
    for value, subset_targets in best_splits.items():
        subset_data = data[data[:, best_feature] == value]
        new_attributes = [a for a in attributes if a != best_feature]
        child_node = build(subset_data, subset_targets, new_attributes)
        children[value] = child_node

    default_label = most_common_label(targets)
    return Node(attribute=best_feature, children=children, default_label=default_label)


def predict(tree, data):
    if tree.target is not None:
        return tree.target

    if data[tree.attribute] in tree.children:
        return predict(tree.children[data[tree.attribute]], data)
    else:
        return tree.default_label
