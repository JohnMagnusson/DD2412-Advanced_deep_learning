# This file should provide functions to get datasets in a trivial way

import pickle
import numpy as np
import flagSettings
import tensorflow as tf

def get_data_set():

    if flagSettings.data_set == "cifar-10":

        x_train, y_train, x_test, y_test = load_cifar10(5)
        train_data = tf.data.Dataset.from_tensor_slices((tf.cast(x_train, dtype=tf.float32), tf.keras.utils.to_categorical(y_train, 10)))
        test_data = tf.data.Dataset.from_tensor_slices((tf.cast(x_test, dtype=tf.float32), tf.keras.utils.to_categorical(y_test, 10)))
        return train_data, test_data
    else:
        raise NotImplemented("This dataset is not implemented yet : " + flagSettings.data_set)

def load_cifar10(nrBatches=5):
    training_data = []
    training_labels = []
    for i in range(1, (nrBatches+1)):
        x, y = load_cifar10_batch("data_batch_" + str(i))
        training_data.extend(x)
        training_labels.extend(y)

    test_data, test_labels = load_cifar10_batch("test_batch")

    return np.array(training_data), np.array(training_labels), test_data, np.array(test_labels)

def load_cifar10_batch(batch_name):
    with open("../datasets/cifar-10/" + str(batch_name), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    data = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return data, labels
