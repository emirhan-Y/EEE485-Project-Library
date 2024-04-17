import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time

from cnn.convolutional_layer import convolutional_layer
from cnn.dense_layer import dense_layer
from cnn.cnn import cnn
from cnn.flatten_layer import flatten_layer
from cnn.sigmoid_activation import sigmoid_activation
from cnn.softmax_activation import softmax_activation

rng = np.random.default_rng(1337)
cnn = cnn([convolutional_layer((28, 28), 1, (3, 3), 5, rng),
           sigmoid_activation(),
           flatten_layer((5, 26, 26), (3380, 1)),
           dense_layer(3380, 100, rng),
           sigmoid_activation(),
           dense_layer(100, 10, rng),
           softmax_activation()
           ])

training_dataset_abspath = os.path.abspath("../data/mnist/mnist_train.csv")  # main folder path
training_dataset = pd.read_csv(training_dataset_abspath)
Y_precursor = training_dataset['label'].to_numpy()
X_cols = training_dataset.columns.drop(['label']).to_numpy()
train_X = training_dataset[X_cols].to_numpy(dtype=np.float64)
train_X = train_X.reshape(train_X.shape[0], 1, 28, 28)
train_Y = np.zeros((len(train_X), 10), dtype=np.float64)
train_Y = train_Y.reshape(train_Y.shape[0], train_Y.shape[1], 1)
for i in range(len(train_Y)):
    train_Y[i][Y_precursor[i]][0] = 1

test_dataset_abspath = os.path.abspath("../data/mnist/mnist_test.csv")  # main folder path
test_dataset = pd.read_csv(test_dataset_abspath)
Y_precursor = test_dataset['label'].to_numpy()
X_cols = test_dataset.columns.drop(['label']).to_numpy()
test_X = test_dataset[X_cols].to_numpy(dtype=np.float64)
test_X = test_X.reshape(test_X.shape[0], 1, 28, 28)
test_Y = np.zeros((len(test_X), 10), dtype=np.float64)
test_Y = test_Y.reshape(test_Y.shape[0], test_Y.shape[1], 1)
for i in range(len(test_Y)):
    test_Y[i][Y_precursor[i]][0] = 1


def process_batch(batch):
    return [cnn.fwd_prop(x) for x in batch]


if __name__ == '__main__':
    start = time.time()
    '''chunk_size = 100  # Adjust based on experimentation
    batches = [train_X[i:i + chunk_size] for i in range(0, len(train_X), chunk_size)]

    with Pool(processes=6) as pool:
        Y_hat = pool.map(process_batch, batches)
    elapsed = time.time() - start
    print(f'Using Pool, time elapsed: {elapsed}')'''

    epsilon = 1e-9  # small number to stabilize log
    stop = False
    epoch = 1
    max_epochs = 100  # for example
    while not stop and epoch <= max_epochs:
        E = 0
        train_c = 0
        dE_dY = np.zeros_like(train_Y[0])
        for i in range(len(train_X)):
            cur_X = train_X[i].reshape(1, 28, 28)
            cur_Y = train_Y[i]
            Y_hat = cnn.fwd_prop(cur_X)
            if cur_Y.argmax() == Y_hat.argmax():
                train_c += 1
            E -= np.sum(cur_Y * np.log(Y_hat + epsilon))
            dE_dY -= cur_Y / (Y_hat + epsilon)
        print(f'Loss: {E}')
        print(f'Training accuracy: {train_c / len(train_X)}')
        E /= len(train_X)
        dE_dY /= len(train_X)
        cnn.bck_prop(dE_dY, 1)
        print(f'epoch: {epoch} complete')
        epoch += 1

    print('foo')
