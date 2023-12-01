import numpy as np
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


def normalize_min_max(data):
    data = data.astype(float)
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min)
    return data


def normalize_min_max_v2(data, new_min, new_max):
    data = data.astype(float)
    data_min = np.min(data)
    data_max = np.max(data)
    # Normalize the data to 0-1 range
    normalized_data = (data - data_min) / (data_max - data_min)
    # Scale to new_min-new_max range
    scaled_data = normalized_data * (new_max - new_min) + new_min
    return scaled_data

def standardize(data):
    return (data - np.mean(data)) / np.std(data)