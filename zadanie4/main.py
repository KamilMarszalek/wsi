from zadanie4.id3 import *
import pandas as pd
import numpy as np
from typing import Tuple


DATA_SETS = ["agaricus-lepiota.data", "breast-cancer.data"]
DATA = DATA_SETS[0]
TRAIN_SIZE = 0.6


def test_model(
    dataset: str = DATA,
    train_size: float = TRAIN_SIZE,
    class_index: int = 0,
    begin: int = 1,
    end: Optional[int] = None,
    row_percentage: float = 1,
) -> Tuple[float, np.ndarray]:
    data = pd.read_csv(dataset, header=0)
    shuffled_data = data.sample(
        frac=row_percentage, random_state=np.random.randint(0, 10000)
    ).reset_index(drop=True)
    split = int(train_size * len(shuffled_data))
    train_data = shuffled_data.iloc[:split]
    test_data = shuffled_data.iloc[split:]
    attr_indices = list(
        range(begin, end) if end is not None else range(begin, len(data.columns))
    )
    if class_index in attr_indices:
        attr_indices.remove(class_index)
    data = train_data.iloc[:, attr_indices].values
    targets = train_data.iloc[:, class_index].values
    attributes = list(range(data.shape[1]))
    tree = build(data, targets, attributes)
    unique_targets = np.unique(
        np.concatenate(
            (
                train_data.iloc[:, class_index].values,
                test_data.iloc[:, class_index].values,
            )
        )
    )
    map_of_targets = {k: v for v, k in enumerate(unique_targets)}
    confusion_matrix = np.zeros((2, 2), dtype=int)
    count_positive = 0
    for i in range(len(test_data)):
        data = test_data.iloc[i, attr_indices].values
        target = test_data.iloc[i, class_index]
        prediction = predict(tree, data)
        if prediction is not None and target is not None:
            if prediction == target:
                count_positive += 1
            confusion_matrix[map_of_targets[target], map_of_targets[prediction]] += 1
    return (count_positive / len(test_data), confusion_matrix)


if __name__ == "__main__":
    results = []
    matrices = []
    for i in range(100):
        test_accuracy = test_model()
        results.append(test_accuracy[0])
        matrices.append(test_accuracy[1])
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
