import os
import numpy as np
import cv2
import time

from models.cnn import cnn
from models.cnn import convolutional_layer
from models.cnn import dense_layer
from models.cnn import max_pooling_layer
from models.cnn import flatten_layer

from help import load_signature_data
import matplotlib.pyplot as plt
from help import draw_confusion_matrix


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
               dense_layer(100, 4, rng, 'softmax')], 'cross_entropy')

# Define parameters
epochs = 32
eta = 0.0001  # Learning rate
loss_array = np.zeros(train_X.shape[0])
training_loss_array = []
training_accuracy_array = []
test_loss_array = []
test_accuracy_array = []

# Training loop
for epoch in range(epochs):
    start = time.time()
    training_accuracy = 0
    total_loss = 0

    perm = np.random.permutation(train_X.shape[0])
    train_X = train_X[perm]
    train_Y = train_Y[perm]

    for i in range(train_X.shape[0]):
        if i % (train_X.shape[0] // 10) == 0 and epoch != 0:
            training_loss_array.append(np.sum(loss_array)/train_X.shape[0])

        x = train_X[i]  # Current sample features
        y = train_Y[i]  # Current sample label

        # Forward pass
        predictions = network.fwd_prop(x)  # Reshape x to match input dimensions
        if np.argmax(y) == np.argmax(predictions):
            training_accuracy += 1
        # Compute loss
        loss = cross_entropy_loss(predictions, y.reshape(-1, 1))  # Reshape y to match output dimensions
        loss_array[i] = loss
        total_loss += loss

        # Compute derivatives
        dL_dY = cross_entropy_derivative(predictions, y.reshape(-1, 1))

        # Backward pass with the averaged gradients
        network.bck_prop(dL_dY, eta)

    average_loss = total_loss / train_X.shape[0]
    training_accuracy /= train_X.shape[0]
    training_accuracy_array.append(training_accuracy)
    print(
        f'Epoch {epoch + 1}: Average training loss: {average_loss:.4f}, Training accuracy = {training_accuracy:.4f}, '
        f'Time = {(time.time() - start):.4f} seconds')

    test_accuracy = 0
    total_loss = 0

    for i in range(test_X.shape[0]):
        x = test_X[i]  # Current sample features
        y = test_Y[i]  # Current sample label

        # Forward pass
        predictions = network.fwd_prop(x)  # Reshape x to match input dimensions
        if np.argmax(y) == np.argmax(predictions):
            test_accuracy += 1
        # Compute loss
        loss = cross_entropy_loss(predictions, y.reshape(-1, 1))  # Reshape y to match output dimensions
        total_loss += loss

    average_loss = total_loss / test_X.shape[0]
    test_loss_array.append(average_loss)
    test_accuracy /= test_X.shape[0]
    test_accuracy_array.append(test_accuracy)
    print(f'Epoch {epoch + 1} TEST: Average test loss: {average_loss:.4f}, Test accuracy = {test_accuracy:.4f}')

training_loss_array.append(np.sum(loss_array)/ train_X.shape[0])
training_loss_array = np.array(training_loss_array)
test_loss_array = np.array(test_loss_array)

# loss graph
index = np.arange(len(training_loss_array))
index_scaled = index / 10
index_pro_scaled = np.arange(len(training_loss_array) / 10 + 1)
plt.figure(figsize=(8, 4))
plt.plot(index_scaled, training_loss_array, linestyle='-', color='r',  label="Average Training Loss")
plt.plot(test_loss_array, linestyle='-', color='b',  label="Average Test Loss")
plt.title('Average loss after each epoch', fontsize=32)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(loc="upper right")
plt.grid(True)
plt.xticks(ticks=index_pro_scaled, labels=index_pro_scaled.astype(int) + 1)
plt.show()

# accuracy graph
ticks = np.arange(len(training_accuracy_array))
plt.figure(figsize=(8, 4))
plt.plot(training_accuracy_array, linestyle='-', color='r',  label="Training Accuracy")
plt.plot(test_accuracy_array, linestyle='-', color='b',  label="Test Accuracy")
plt.title('Accuracy after each epoch', fontsize=32)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc="lower right")
plt.grid(True)
# Setting the x-axis ticks to the original index values
plt.xticks(ticks=ticks, labels=ticks.astype(int) + 1)
plt.show()

# confusion matrix
prediction_array = []
for i in range(test_X.shape[0]):
    x = test_X[i]  # Current sample features
    y = test_Y[i]  # Current sample label

    # Forward pass
    predictions = network.fwd_prop(x)  # Reshape x to match input dimensions
    prediction_array.append(predictions)
label_array = np.argmax(test_Y.reshape(test_Y.shape[0], test_Y.shape[2]), axis=1)
prediction_array = np.argmax(prediction_array, axis=1)
draw_confusion_matrix(label_array, prediction_array, 'test')
print('foo')
