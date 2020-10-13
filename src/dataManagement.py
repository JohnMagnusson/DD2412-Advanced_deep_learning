# This file should provide functions to get datasets in a trivial way

import pickle
import numpy as np
import flagSettings

def get_data_set():

    if flagSettings.data_set == "cifar-10":
        return load_cifar10(5)
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
    with open("../../datasets/cifar-10/" + str(batch_name), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    data = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return data, labels
