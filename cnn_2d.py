import json
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from data_utils import normalize_min_max_v2, prepare_data, shuffle_data

RAND_SEED = 1
np.random.seed(RAND_SEED)
tf.random.set_seed(RAND_SEED)
random.seed(RAND_SEED)


def train_cnn(train_set, test_set, window_size, overlap, exp_name):
    X_train, y_train = prepare_data(train_set)
    X_test, y_test = prepare_data(test_set)

    X_train = normalize_min_max_v2(X_train, 0, 1)
    X_test = normalize_min_max_v2(X_test, 0, 1)

    # reshape only when grayscale
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    DROPOUT_RATE = 0.15
    KERNEL_SIZE = (5, 5)
    L2_REGULARIZATION = 0.0001

    model = Sequential(
        [
            Conv2D(
                20,
                KERNEL_SIZE,
                padding="same",
                kernel_regularizer=l2(L2_REGULARIZATION),
                input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
            ),
            Activation("relu"),
            MaxPooling2D(
                (2, 2),
            ),
            # Dropout(DROPOUT_RATE),
            Conv2D(
                40,
                KERNEL_SIZE,
                padding="same",
                kernel_regularizer=l2(L2_REGULARIZATION),
            ),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            # Dropout(DROPOUT_RATE),
            # Conv2D(
            #     80,
            #     KERNEL_SIZE,
            #     padding="same",
            #     kernel_regularizer=l2(L2_REGULARIZATION),
            # ),
            # Activation("relu"),
            # MaxPooling2D((2, 2)),
            # # Dropout(DROPOUT_RATE),
            Flatten(),
            Dense(
                512,
                # kernel_regularizer=l2(L2_REGULARIZATION)
            ),
            Activation("relu"),
            Dropout(DROPOUT_RATE),
            Dense(20, activation="softmax"),
        ]
    )
    model.summary()

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Define the path where you want to save the best checkpoint
    checkpoint_filepath = (
        f"results/cnn2d_models/cnn2d_{window_size}_{overlap}_{exp_name}.h5"
    )
    # Define the ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",  # Choose the metric to monitor (e.g., validation loss)
        mode="max",  # 'min' for metrics like validation loss, 'max' for accuracy, etc.
        save_best_only=True,  # Save only the best model checkpoint
        save_weights_only=False,  # Save the entire model, not just weights
        verbose=1,  # Print messages about checkpoint saving
    )

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        monitor="loss", min_delta=0.01, factor=0.6, patience=1, min_lr=0.0000001
    )

    metrics = {
        "train_losses": [],
        "test_losses": [],
        "train_accuracies": [],
        "test_accuracies": [],
    }

    for epoch in range(500):
        X_train, y_train = shuffle_data(X_train, y_train)

        history = model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=40,
            verbose=0,
            callbacks=[lr_scheduler, model_checkpoint],
            validation_data=(X_test, y_test),
        )  # Train for one epoch at a time

        # Store training metrics
        metrics["train_losses"].append(history.history["loss"][0])
        metrics["train_accuracies"].append(history.history["accuracy"][0])

        # Store test/validation metrics
        metrics["test_losses"].append(history.history["val_loss"][0])
        metrics["test_accuracies"].append(history.history["val_accuracy"][0])

    # save metrics in json file
    with open(
        f"results/cnn2d_models/cnn2d_{window_size}_{overlap}_{exp_name}.json", "w"
    ) as fp:
        json.dump(metrics, fp)


NPY_DATA_DIR = "npy_datasets"

spectogram_map = {
    256: [8, 64, 128, 250],
    512: [0, 256, 511],
}

window_size = 256
overlap = 8

for window_size, overlap_list in spectogram_map.items():
    for overlap in overlap_list:
        # Load the dataset
        test_dataset = np.load(
            f"{NPY_DATA_DIR}/test_{window_size}_{overlap}_png.npy", allow_pickle=True
        )
        retest_dataset = np.load(
            f"{NPY_DATA_DIR}/retest_{window_size}_{overlap}_png.npy", allow_pickle=True
        )

        test_retest_metrics = train_cnn(
            test_dataset, retest_dataset, window_size, overlap, exp_name="test_retest"
        )
        retest_test_metrics = train_cnn(
            retest_dataset, test_dataset, window_size, overlap, exp_name="retest_test"
        )
