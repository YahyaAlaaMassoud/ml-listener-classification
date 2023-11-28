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
    GlobalAveragePooling2D,
    concatenate
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

RAND_SEED = 1
np.random.seed(RAND_SEED)
tf.random.set_seed(RAND_SEED)
random.seed(RAND_SEED)

from data_utils import flatten_data, normalize_min_max, shuffle_data, normalize_min_max_v2, one_hot_encode


def train_cnn(train_set, test_set):
    X_train, y_train = train_set[:, 0], train_set[:, 1]
    X_test, y_test = test_set[:, 0], test_set[:, 1]
    
    X_train = np.array([np.array(x) for x in X_train])
    X_test = np.array([np.array(x) for x in X_test])
    
    y_train = np.array([one_hot_encode(x) for x in y_train])
    y_test = np.array([one_hot_encode(x) for x in y_test])
    
    print("X_train.shape", X_train.shape)
    print("y_train.shape", y_train.shape)
    print("X_test.shape", X_test.shape)
    print("y_test.shape", y_test.shape)

    X_train = normalize_min_max_v2(X_train, 0, 1)
    X_test = normalize_min_max_v2(X_test, 0, 1)

    print("X_train.min()", X_train.min(), "X_train.max()", X_train.max())
    print("X_test.min()", X_test.min(), "X_test.max()", X_test.max())
    print("y_train.min()", y_train.min(), "y_train.max()", y_train.max())
    print("y_test.min()", y_test.min(), "y_test.max()", y_test.max())
    
    # reshape only when grayscale
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_test.shape[3], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1)
    
    print("X_train.shape", X_train.shape)
    print("X_test.shape", X_test.shape)
    
    DROPOUT_RATE = 0.3
    KERNEL_SIZE = (3,3)
    L2_REGULARIZATION = 0.00001
    
    # Define four input layers
    input_layers = [Input(shape=(32, 32, 1)) for _ in range(X_train.shape[1])]

    # Define a shared CNN layer
    def create_conv_layers(input_img):
        x = Conv2D(8, KERNEL_SIZE, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(input_img)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(16, KERNEL_SIZE, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, KERNEL_SIZE, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
        x = MaxPooling2D((2, 2))(x)
        return x

    # Apply the shared layers to each input
    processed_inputs = [create_conv_layers(inp) for inp in input_layers]

    # Concatenate the outputs of the CNN layers
    concatenated = concatenate([Flatten()(inp) for inp in processed_inputs])

    # Add dense layers for classification
    x = Dense(512, activation='relu')(concatenated)
    x = Dropout(DROPOUT_RATE)(x)
    
    output = Dense(20, activation='softmax')(x)  # 20 classes

    # Create the model
    model = Model(inputs=input_layers, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    best_test_acc = 0  # Initialize the best test accuracy
    best_test_loss = float("inf")  # Initialize the best test loss as infinity
    
    X_test = [X_test[:, i, :, :, :] for i in range(X_test.shape[1])]
    
    print(X_train[0].shape)

    for epoch in range(1000):
        X_train, y_train = shuffle_data(X_train, y_train)
        
        model.fit(
            [X_train[:, i] for i in range(X_train.shape[1])], y_train, epochs=1, batch_size=40, verbose=0,
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

window_size = 512

# Load the dataset
test_dataset = np.load(
    f"{NPY_DATA_DIR}/test_{window_size}_multires.npy", allow_pickle=True
)
retest_dataset = np.load(
    f"{NPY_DATA_DIR}/retest_{window_size}_multires.npy", allow_pickle=True
)

print("test_dataset.shape", test_dataset.shape)
print("retest_dataset.shape", retest_dataset.shape)

test_retest_metrics = train_cnn(test_dataset, retest_dataset)
# retest_test_metrics = train_cnn(retest_dataset, test_dataset)
