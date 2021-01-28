import numpy as np
import sys

try:
    import tensorflow as tf
except ModuleNotFoundError:
    print('Can not download mnist dataset. tensorflow not installed')
    sys.exit(-1)

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

old_save = np.save

np.save = lambda x,y: old_save(y, x)


path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
  train_examples = data['x_train']
  train_labels = data['y_train']
  test_examples = data['x_test']
  test_labels = data['y_test']

  np.save(train_examples, 'x_train')
  np.save(train_labels, 'y_train')
  np.save(test_examples, 'x_test')
  np.save(test_labels, 'y_test') 
