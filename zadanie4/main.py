from id3 import *
import pandas as pd


DATA_SETS = ["agaricus-lepiota.data", "breast-cancer.data"]
DATA = DATA_SETS[1]


def test_model():
    data = pd.read_csv(DATA, header=None)
    shuffled_data = data.sample(
        frac=1, random_state=np.random.randint(0, 10000)
    ).reset_index(drop=True)
    split = int(0.6 * len(data))
    train_data = shuffled_data.iloc[split:]
    test_data = shuffled_data.iloc[:split]
    features = train_data.iloc[:, 1:].values
    targets = train_data.iloc[:, 0].values
    tree = build(features, targets)
    map_of_targets = {k: v for v, k in enumerate(np.unique(targets))}
    confusion_matrix = np.zeros((2, 2))
    count_positive = 0
    for i in range(len(test_data)):
        features = test_data.iloc[i, 1:].values
        target = test_data.iloc[i, 0]
        prediction = predict(tree, features)
        if prediction is not None and target is not None:
            if prediction == target:
                count_positive += 1
            confusion_matrix[map_of_targets[target], map_of_targets[prediction]] += 1
    return (count_positive / len(test_data), confusion_matrix)


if __name__ == "__main__":
    data = pd.read_csv(DATA, header=None)
    print(data)
    results = []
    for _ in range(10000):
        test_accuracy = test_model()
        # print("Accuracy:", test_accuracy[0])
        results.append(test_accuracy[0])
        # print("Confusion matrix:\n", test_accuracy[1])
    mean_accuracy = np.mean(results)
    min_accuracy = np.min(results)
    max_accuracy = np.max(results)
    std_accuracy = np.std(results)
    print("Mean accuracy:", mean_accuracy)
    print("Min accuracy:", min_accuracy)
    print("Max accuracy:", max_accuracy)
    print("Std accuracy:", std_accuracy)
