# This file should provide functions to get datasets in a trivial way

import pickle
import numpy as np
import flagSettings

def getDataSet():

    if flagSettings.dataSet == "cifar-10":
        return load_cifar10(5)
    else:
        raise NotImplemented("This dataset is not implemented yet : " + flagSettings.dataSet)

def load_cifar10(nrBatches=5):
    trainingData = []
    trainingLabels = []
    for i in range(1, (nrBatches+1)):
        x, y = load_cifar10_batch("data_batch_" + str(i))
        trainingData.extend(x)
        trainingLabels.extend(y)

    testData, testLabels = load_cifar10_batch("test_batch")

    return np.array(trainingData), np.array(trainingLabels), testData, np.array(testLabels)

def load_cifar10_batch(batchName):
    with open("../datasets/cifar-10/" + str(batchName), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    data = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return data, labels
