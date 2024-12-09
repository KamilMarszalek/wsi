import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, target=None, children=None):
        self.feature = feature
        self.target = target
        self.children = children if children is not None else {}


def entropy(targets):
    counts = Counter(targets).values()
    probabilities = [c / len(targets) for c in counts]
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


def most_common_label(targets):
    targets = [t for t in targets if t is not None]
    if not targets:
        return None
    return Counter(targets).most_common(1)[0][0]


def build(features, targets):
    if len(set(targets)) == 1:
        return Node(target=targets[0])
    if len(features) == 0 :
        return Node(target=Counter(targets).most_common(1)[0][0])

    current_entropy = entropy(targets)
    best_gain = 0
    best_feature = None
    best_splits = None

    n_features = features.shape[1]

    for feature in range(n_features):
        values = np.unique(features[:, feature])
        splits = {}
        for value in values:
            subset_indices = features[:, feature] == value
            splits[value] = targets[subset_indices]

        weighted_entropy = sum(
            (len(subset) / len(targets)) * entropy(subset) for subset in splits.values()
        )
        gain = current_entropy - weighted_entropy

        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_splits = splits

    if best_gain == 0:
        return Node(target=most_common_label(targets))

    children = {}
    for value, subset_targets in best_splits.items():
        subset_features = features[features[:, best_feature] == value]
        child_node = build(subset_features, subset_targets)
        children[value] = child_node

    return Node(feature=best_feature, children=children)


def predict(tree, features):
    if tree.target is not None:
        return tree.target

    if features[tree.feature] in tree.children:
        return predict(tree.children[features[tree.feature]], features)
    else:
        return most_common_label([child.target for child in tree.children.values()])
