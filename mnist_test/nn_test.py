import os
import numpy as np
import pandas as pd
import time

from dense_layer import dense_layer
from nn import nn


def cross_entropy_loss(prediction, targets):
    epsilon = 1e-9
    prediction = np.clip(prediction, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(prediction))


def cross_entropy_derivative(predictions, targets):
    return predictions - targets


rng = np.random.default_rng(420)
network = nn([dense_layer(784, 100, rng, 'relu'),
              dense_layer(100, 10, rng, 'softmax')], 'cross_entropy')

training_dataset_abspath = os.path.abspath("./mnist_train.csv")  # main folder path
training_dataset = pd.read_csv(training_dataset_abspath)
Y_precursor = training_dataset['label'].to_numpy()
X_cols = training_dataset.columns.drop(['label']).to_numpy()
train_X = training_dataset[X_cols].to_numpy(dtype=np.float64)
train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1]) / 255
train_Y = np.zeros((len(train_X), 10), dtype=np.float64)
train_Y = train_Y.reshape(train_Y.shape[0], train_Y.shape[1], 1)
for i in range(len(train_Y)):
    train_Y[i][Y_precursor[i]][0] = 1

test_dataset_abspath = os.path.abspath("./mnist_test.csv")  # main folder path
test_dataset = pd.read_csv(test_dataset_abspath)
Y_precursor = test_dataset['label'].to_numpy()
X_cols = test_dataset.columns.drop(['label']).to_numpy()
test_X = test_dataset[X_cols].to_numpy(dtype=np.float64)
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1]) / 255
test_Y = np.zeros((len(test_X), 10), dtype=np.float64)
test_Y = test_Y.reshape(test_Y.shape[0], test_Y.shape[1], 1)
for i in range(len(test_Y)):
    test_Y[i][Y_precursor[i]][0] = 1

# Define parameters
epochs = 10
eta = 0.01  # Learning rate

# Training loop
for epoch in range(epochs):
    accuracy = 0
    total_loss = 0

    for i in range(train_X.shape[0]):
        x = train_X[i]  # Current sample features
        y = train_Y[i]  # Current sample label

        # Forward pass
        predictions = network.fwd_prop(x.reshape(-1, 1))  # Reshape x to match input dimensions
        if np.argmax(y) == np.argmax(predictions):
            accuracy += 1
        # Compute loss
        loss = cross_entropy_loss(predictions, y.reshape(-1, 1))  # Reshape y to match output dimensions
        total_loss += loss

        # Compute derivatives
        dL_dY = cross_entropy_derivative(predictions, y.reshape(-1, 1))

        # Backward pass with the averaged gradients
        network.bck_prop(dL_dY, eta)

    average_loss = total_loss / train_X.shape[0]
    accuracy /= train_X.shape[0]
    print(f'Epoch {epoch + 1}: Average Loss: {average_loss:.4f}, Accuracy = {accuracy:.4f}')

print('foo')
