import json

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.utils import shuffle


def one_hot_encode(number, num_labels=20):
    # Ensure the number is in the valid range
    if 0 <= number < num_labels:
        # Create an array of zeros
        one_hot_vector = np.zeros(num_labels, dtype=int)

        # Set the appropriate element to 1
        one_hot_vector[number] = 1

        return one_hot_vector
    else:
        raise ValueError("Number out of range")


def prepare_data(dataset):
    X_train, y_train = dataset[:, 0], dataset[:, 1]

    X_train = np.array([np.array(x) for x in X_train])
    y_train = np.array([one_hot_encode(label) for label in y_train])

    return X_train, y_train


def flatten_data(data):
    # Flatten the 3D features into 1D
    data_flat = data.reshape(data.shape[0], -1)
    return data_flat


def shuffle_data(X, y):
    X_shuffled, y_shuffled = shuffle(X, y, random_state=0)
    return X_shuffled, y_shuffled


def train_svm(train_set, test_set):
    X_train, y_train = prepare_data(train_set)
    X_test, y_test = prepare_data(test_set)

    print("X_train.shape", X_train.shape)
    print("y_train.shape", y_train.shape)
    print("X_test.shape", X_test.shape)
    print("y_test.shape", y_test.shape)

    X_train_flat = flatten_data(X_train)
    X_test_flat = flatten_data(X_test)

    # Convert one-hot encoded labels to class indices
    y_train_idx = np.argmax(y_train, axis=1)
    y_test_idx = np.argmax(y_test, axis=1)

    X_train_flat, y_train_idx = shuffle_data(X_train_flat, y_train_idx)

    # Linear Kernel
    svm_linear = SVC(kernel="linear")

    # Polynomial Kernel
    svm_poly = SVC(
        kernel="poly", degree=3
    )  # degree is a hyperparameter for polynomial kernel

    # RBF Kernel
    svm_rbf = SVC(kernel="rbf", gamma="scale")

    svm_linear.fit(X_train_flat, y_train_idx)
    svm_poly.fit(X_train_flat, y_train_idx)
    svm_rbf.fit(X_train_flat, y_train_idx)

    # Making predictions and evaluating accuracy for each model
    y_pred_linear = svm_linear.predict(X_test_flat)
    y_pred_poly = svm_poly.predict(X_test_flat)
    y_pred_rbf = svm_rbf.predict(X_test_flat)

    accuracy_linear = accuracy_score(y_test_idx, y_pred_linear)
    accuracy_poly = accuracy_score(y_test_idx, y_pred_poly)
    accuracy_rbf = accuracy_score(y_test_idx, y_pred_rbf)

    print("-" * 50)
    print(f"Linear Kernel Accuracy: {accuracy_linear}")
    print(f"Polynomial Kernel Accuracy: {accuracy_poly}")
    print(f"RBF Kernel Accuracy: {accuracy_rbf}")
    print("-" * 50)

    # save results in dict
    metrics = {
        "linear": accuracy_linear,
        "poly": accuracy_poly,
        "rbf": accuracy_rbf,
    }

    return metrics


NPY_DATA_DIR = "npy_datasets"
RESULTS_DIR = "results"

spectogram_map = {
    256: [8, 64, 128, 250],
    512: [0, 256, 511],
}

results = {}

for window_size, overlaps in spectogram_map.items():
    for overlap in overlaps:
        print(f"window_size: {window_size}, overlap: {overlap}")

        # Load the dataset
        test_dataset = np.load(
            f"{NPY_DATA_DIR}/test_{window_size}_{overlap}.npy", allow_pickle=True
        )
        retest_dataset = np.load(
            f"{NPY_DATA_DIR}/retest_{window_size}_{overlap}.npy", allow_pickle=True
        )

        test_retest_metrics = train_svm(test_dataset, retest_dataset)
        retest_test_metrics = train_svm(retest_dataset, test_dataset)

        results[f"{window_size}_{overlap}"] = {
            "test_retest": test_retest_metrics,
            "retest_test": retest_test_metrics,
        }

# save results as pretty printed json
with open(f"{RESULTS_DIR}/svm_results.json", "w") as f:
    json.dump(results, f, indent=4)
