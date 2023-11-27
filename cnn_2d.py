import json
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    Lambda,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2

RAND_SEED = 1
np.random.seed(RAND_SEED)
tf.random.set_seed(RAND_SEED)
random.seed(RAND_SEED)

from data_utils import flatten_data, normalize_min_max, prepare_data, shuffle_data, normalize_min_max_v2


def lrn(input, radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(input, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)


def train_cnn(train_set, test_set):
    X_train, y_train = prepare_data(train_set)
    X_test, y_test = prepare_data(test_set)

    X_train = normalize_min_max_v2(X_train, 0, 1)
    X_test = normalize_min_max_v2(X_test, 0, 1)

    print("X_train.min()", X_train.min(), "X_train.max()", X_train.max())
    print("X_test.min()", X_test.min(), "X_test.max()", X_test.max())
    print("y_train.min()", y_train.min(), "y_train.max()", y_train.max())
    print("y_test.min()", y_test.min(), "y_test.max()", y_test.max())

    print("X_train.shape", X_train.shape)
    print("y_train.shape", y_train.shape)
    print("X_test.shape", X_test.shape)
    print("y_test.shape", y_test.shape)

    X_train, y_train = shuffle_data(X_train, y_train)
    
    # reshape only when grayscale
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    print("X_train.shape", X_train.shape)
    print("y_train.shape", y_train.shape)
    print("X_test.shape", X_test.shape)
    print("y_test.shape", y_test.shape)

    DROPOUT_RATE = 0.5
    KERNEL_SIZE = (7,7)
    L2_REGULARIZATION = 0.00001

    model = Sequential(
        [
            Conv2D(
                20,
                KERNEL_SIZE,
                padding="same",
                activation="relu",
                kernel_regularizer=l2(L2_REGULARIZATION),
                input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
            ),
            MaxPooling2D(
                (2, 2), 
            ),
            
            Conv2D(
                80, 
                KERNEL_SIZE, 
                activation="relu", 
                padding="same",
                kernel_regularizer=l2(L2_REGULARIZATION),
            ),
            MaxPooling2D(
                (2, 2)
            ),
            
            Flatten(),
            
            Dense(512, activation="relu"),#, kernel_regularizer=l2(L2_REGULARIZATION)),
            Dropout(DROPOUT_RATE),

            Dense(20, activation="softmax"),
        ]
    )
    model.summary()

    model.compile(
        optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"]
    )
    
    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='loss', min_delta=0.01, factor=0.6, patience=1, min_lr=0.0000001)

    best_test_acc = 0  # Initialize the best test accuracy
    best_test_loss = float("inf")  # Initialize the best test loss as infinity

    for epoch in range(1000):
        X_train, y_train = shuffle_data(X_train, y_train)
        
        model.fit(
            X_train, y_train, epochs=1, batch_size=40, verbose=1,
            # callbacks=[lr_scheduler]
        )  # Train for one epoch at a time

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        # Compare and store best test loss and accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch_acc = epoch
            best_test_loss = test_loss
            best_epoch_loss = epoch

        if epoch % 20 == 0:
            # After training, print the best test loss and accuracy
            try:
                print("current epoch: ", epoch)
                print(f"Best Test Accuracy: {best_test_acc} at epoch {best_epoch_acc}")
                print(f"Lowest Test Loss: {best_test_loss} at epoch {best_epoch_loss}")
            except:
                pass


NPY_DATA_DIR = "npy_datasets"
RESULTS_DIR = "results"

spectogram_map = {
    256: [8, 64, 128, 250],
    512: [0, 256, 511],
}

window_size = 256
overlap = 250

# Load the dataset
test_dataset = np.load(
    f"{NPY_DATA_DIR}/test_{window_size}_{overlap}_png.npy", allow_pickle=True
)
retest_dataset = np.load(
    f"{NPY_DATA_DIR}/retest_{window_size}_{overlap}_png.npy", allow_pickle=True
)

test_retest_metrics = train_cnn(test_dataset, retest_dataset)
# retest_test_metrics = train_cnn(retest_dataset, test_dataset)
