from id3 import *
import pandas as pd

DATA = "agaricus-lepiota.data"

if __name__ == "__main__":
    data = pd.read_csv(DATA, header=None)
    print(data)
    shuffled_data = data.sample(frac=1, random_state=np.random.seed()).reset_index(
        drop=True
    )
    split = int(0.6 * len(data))
    train_data = shuffled_data.iloc[split:]
    test_data = shuffled_data.iloc[:split]
    features = train_data.iloc[:, 1:].values
    targets = train_data.iloc[:, 0].values
    # targets = np.where(targets == "no-recurrence-events", 0, 1)

    print(targets)
    print(features)
    tree = build(features, targets)
    print(tree)
    count_positive = 0
    for i in range(len(test_data)):
        features = test_data.iloc[i, 1:].values
        target = test_data.iloc[i, 0]
        prediction = predict(tree, features)
        # prediction = "no-recurrence-events" if prediction == 0 else "recurrence-events"
        print(f"Prediction: {prediction}, Actual: {target}")
        if prediction == target:
            count_positive += 1
    print(f"Accuracy: {count_positive / len(test_data)}")
