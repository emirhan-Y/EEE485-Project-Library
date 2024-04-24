import os
import numpy as np
import cv2

from models.cnn import cnn
from models.cnn import convolutional_layer
from models.cnn import dense_layer
from models.cnn import max_pooling_layer
from models.cnn import flatten_layer

from help import load_signature_data


def cross_entropy_loss(prediction, targets):
    epsilon = 1e-9
    prediction = np.clip(prediction, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(prediction))


def cross_entropy_derivative(predictions, targets):
    return predictions - targets

rng = np.random.default_rng(420)
data_folder = os.path.abspath('_data/final')

train_X, train_Y, test_X, test_Y = load_signature_data(data_folder, 420, test_percentage=0.2, hot_encode=True)
network = cnn([convolutional_layer((50, 126), 1, (5, 5), 8, rng, 'relu'),
               max_pooling_layer((46, 122), (4, 4)),
               flatten_layer((8, 12, 31), 8 * 12 * 31),
               dense_layer(8 * 12 * 31, 100, rng, 'relu'),
               dense_layer(100, 2, rng, 'softmax')], 'cross_entropy')

# Define parameters
epochs = 1000
eta = 0.001  # Learning rate

# Training loop
for epoch in range(epochs):
    accuracy = 0
    total_loss = 0

    perm = np.random.permutation(train_X.shape[0])
    train_X = train_X[perm]
    train_Y = train_Y[perm]

    for i in range(train_X.shape[0]):
        x = train_X[i]  # Current sample features
        y = train_Y[i]  # Current sample label

        # Forward pass
        predictions = network.fwd_prop(x)  # Reshape x to match input dimensions
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
