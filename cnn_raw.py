import random
import numpy as np
import tensorflow as tf
import json

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Activation,
    Conv1D, 
    MaxPooling1D,
    Dropout
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from data_utils import prepare_data, shuffle_data


RAND_SEED = 1
np.random.seed(RAND_SEED)
tf.random.set_seed(RAND_SEED)
random.seed(RAND_SEED)


def train_cnn(train_set, test_set, exp_name):
    X_train, y_train = prepare_data(train_set)
    X_test, y_test = prepare_data(test_set)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    KERNEL_SIZE = 32
    L2_REGULARIZATION = 0.0001

    model = Sequential(
        [
            Conv1D(
                8,
                KERNEL_SIZE,
                padding="same",
                kernel_regularizer=l2(L2_REGULARIZATION),
                input_shape=(X_train.shape[1], 1),  # Input signal size is 1300, and 1 channel (1D)
            ),
            Activation('relu'),
            MaxPooling1D(
                2,  # Pooling size
            ),

            Conv1D(
                16,
                KERNEL_SIZE,
                padding="same",
                kernel_regularizer=l2(L2_REGULARIZATION),
            ),
            Activation('relu'),
            MaxPooling1D(
                2,  # Pooling size
            ),

            Flatten(),

            Dense(128),
            Activation('relu'),
            Dropout(0.25),

            Dense(20, activation="softmax"),
        ]
    )
    model.summary()

    model.compile(
        optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Define the path where you want to save the best checkpoint
    checkpoint_filepath = f'results/cnn1d_models/cnn1d_raw_{exp_name}.h5'
    # Define the ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_accuracy',  # Choose the metric to monitor (e.g., validation loss)
        mode='max',  # 'min' for metrics like validation loss, 'max' for accuracy, etc.
        save_best_only=True,  # Save only the best model checkpoint
        save_weights_only=False,  # Save the entire model, not just weights
        verbose=1,  # Print messages about checkpoint saving
    )

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='loss', min_delta=0.01, factor=0.6, patience=1, min_lr=0.0000001)

    metrics = {
        'train_losses': [],
        'test_losses': [],
        'train_accuracies': [],
        'test_accuracies': []
    }

    for epoch in range(500):
        X_train, y_train = shuffle_data(X_train, y_train)
        
        history = model.fit(
            X_train, 
            y_train, 
            epochs=1, 
            batch_size=40, 
            verbose=0,
            callbacks=[
                lr_scheduler, 
                model_checkpoint
            ],
            validation_data=(X_test, y_test)
        )
        
        # Store training metrics
        metrics['train_losses'].append(history.history['loss'][0])
        metrics['train_accuracies'].append(history.history['accuracy'][0])

        # Store test/validation metrics
        metrics['test_losses'].append(history.history['val_loss'][0])
        metrics['test_accuracies'].append(history.history['val_accuracy'][0])
        
    # save metrics in json file
    with open(f'results/cnn1d_models/cnn1d_raw_{exp_name}.json', 'w') as fp:
        json.dump(metrics, fp)


NPY_DATA_DIR = "npy_datasets"

# Load the dataset
test_dataset = np.load(
    f"{NPY_DATA_DIR}/test_aenu.npy", allow_pickle=True
)
retest_dataset = np.load(
    f"{NPY_DATA_DIR}/retest_aenu.npy", allow_pickle=True
)

test_retest_metrics = train_cnn(test_dataset, retest_dataset, exp_name="test_retest")
retest_test_metrics = train_cnn(retest_dataset, test_dataset, exp_name="retest_test")
