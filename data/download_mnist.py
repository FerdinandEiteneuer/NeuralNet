'''
Downloads mnist dataset and saves the data as numpy arrays
'''

import numpy as np
import sys


try:
    import tensorflow as tf
except ModuleNotFoundError:
    print('Can not download mnist dataset. tensorflow not installed')
    sys.exit(-1)

def one_hot(y):
    dim = 10
    matrix = np.eye(dim, dtype=DTYPE)[y]
    matrix = matrix.T
    return matrix

DTYPE = 'float32'
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

def get():
    path = tf.keras.utils.get_file('mnist.npz', DATA_URL)

    with np.load(path) as data:
        # load from disk
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']

        # transform into correct format
        train_labels = one_hot(train_labels)
        test_labels = one_hot(test_labels)
        train_examples = np.transpose(train_examples, (1, 2, 0))
        test_examples = np.transpose(test_examples, (1, 2, 0))

        train_examples = train_examples.astype(DTYPE)
        test_examples = test_examples.astype(DTYPE)


        # save into desired format for easy and fast loading
        np.save('x_train', train_examples)
        np.save('y_train', train_labels)
        np.save('x_test', test_examples)
        np.save('y_test', test_labels)

        return train_examples, train_labels, test_examples, test_labels

if __name__ == '__main__':

    train_examples, train_labels, test_examples, test_labels = get()

    te = train_examples
