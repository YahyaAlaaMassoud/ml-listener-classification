import json
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input,
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
from data_utils import one_hot_encode
from tensorflow.keras.models import Model


RAND_SEED = 1
np.random.seed(RAND_SEED)
tf.random.set_seed(RAND_SEED)
random.seed(RAND_SEED)

from data_utils import flatten_data, normalize_min_max, prepare_data, shuffle_data, normalize_min_max_v2


def lrn(input, radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(input, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)


def train_cnn(train_set, test_set):
    X_train, y_train = train_set[:, 0], train_set[:, 1]
    X_test, y_test = test_set[:, 0], test_set[:, 1]
    
    X_train = np.array([np.array(x) for x in X_train])
    X_test = np.array([np.array(x) for x in X_test])
    
    y_train_regression = np.array([x[0] for x in y_train])
    y_train_classification = np.array([one_hot_encode(x[1]) for x in y_train])
    
    y_test_regression = np.array([x[0] for x in y_test])
    y_test_classification = np.array([one_hot_encode(x[1]) for x in y_test])
    
    # print(y_test_regression)
    # return
    
    print("X_train.shape", X_train.shape)
    print("y_train_regression.shape", y_train_regression.shape)
    print("y_train_classification.shape", y_train_classification.shape)
    print("X_test.shape", X_test.shape)
    print("y_test_regression.shape", y_test_regression.shape)
    print("y_test_classification.shape", y_test_classification.shape)

    X_train = normalize_min_max_v2(X_train, 0, 1)
    X_test = normalize_min_max_v2(X_test, 0, 1)

    print("X_train.min()", X_train.min(), "X_train.max()", X_train.max())
    print("X_test.min()", X_test.min(), "X_test.max()", X_test.max())
    print("y_train_regression.min()", y_train_regression.min(), "y_train_regression.max()", y_train_regression.max())
    print("y_test_regression.min()", y_test_regression.min(), "y_test_regression.max()", y_test_regression.max())
    print("y_train_classification.min()", y_train_classification.min(), "y_train_classification.max()", y_train_classification.max())
    print("y_test_classification.min()", y_test_classification.min(), "y_test_classification.max()", y_test_classification.max())

    # reshape only when grayscale
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    print("X_train.shape", X_train.shape)
    print("X_test.shape", X_test.shape)

    DROPOUT_RATE = 0.3
    KERNEL_SIZE = (7, 7)
    L2_REGULARIZATION = 0.0001

    # Input layer
    input_layer = Input(shape=(32, 32, 1))

    # Convolutional layers
    x = Conv2D(
        20, 
        KERNEL_SIZE, 
        padding="same", 
        activation="relu", 
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(
        40, 
        KERNEL_SIZE, 
        padding="same", 
        activation="relu", 
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten the output
    x = Flatten()(x)

    # Common Dense layer
    x = Dense(512, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)

    # Regression output
    regression_output = Dense(len(y_train_regression[0]), activation='linear', name='regression_output')(x)  # 'linear' activation for regression

    # Classification output
    classification_output = Dense(len(y_train_classification[0]), activation='softmax', name='classification_output')(x)

    # Create the model with two outputs
    model = Model(
        inputs=input_layer, 
        outputs=[
            regression_output, 
            classification_output
        ]
    )
    
    model.summary()

    model.compile(
        optimizer='adam',
        loss={
            'regression_output': 'mean_squared_error',  # MSE for regression
            'classification_output': 'categorical_crossentropy'
        },  # Crossentropy for classification
        metrics={
            'regression_output': ['mean_squared_error'],
            'classification_output': ['accuracy']
        }
    )
    
    best_test_acc = 0  # Initialize the best test accuracy
    best_test_loss = float("inf")  # Initialize the best test loss as infinity

    for epoch in range(1000):
        X_train, y_train = shuffle_data(
            X_train, 
            np.concatenate((y_train_regression, y_train_classification), axis=1)
        )
        y_train_regression = y_train[:, :len(y_train_regression[0])]
        y_train_classification = y_train[:, len(y_train_regression[0]):]
        
        # X_train, y_train_classification = shuffle_data(X_train, y_train_classification)
        
        model.fit(
            X_train, 
            {
                'regression_output': y_train_regression,
                'classification_output': y_train_classification
            },
            epochs=1,
            batch_size=40,
            verbose=0,
        )

        # Evaluate on test set
        test_results = model.evaluate(
            X_test, 
            {
                'regression_output': y_test_regression, 
                'classification_output': y_test_classification
            },
            verbose=0
        )
        
        test_acc = test_results[-1]
        test_loss = test_results[-2]
        
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
RESULTS_DIR = "cnn_results"

spectogram_map = {
    256: [8, 64, 128, 250],
    512: [0, 256, 511],
}

window_size = 256
overlap = 8

# Load the dataset
test_dataset = np.load(
    f"{NPY_DATA_DIR}/test_{window_size}_{overlap}_multitask.npy", allow_pickle=True
)
retest_dataset = np.load(
    f"{NPY_DATA_DIR}/retest_{window_size}_{overlap}_multitask.npy", allow_pickle=True
)

test_retest_metrics = train_cnn(test_dataset, retest_dataset)
# retest_test_metrics = train_cnn(retest_dataset, test_dataset)
