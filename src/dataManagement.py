# This file should provide functions to get datasets in a trivial way
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import flagSettings


def get_data_set(data_set="cifar-10", validation_ratio=0.2):
    if data_set == "cifar-10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_ratio, random_state=1,
                                                          shuffle=False)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    elif data_set == "flowers":
        X = np.load("../../datasets/flowers_processed/flowers_set_x.npy")
        Y = np.load("../../datasets/flowers_processed/flowers_set_y.npy").astype(np.uint8)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=1)

        return (x_train, y_train.reshape(-1, 1)), (x_val, y_val.reshape(-1, 1)), (x_test, y_test.reshape(-1, 1))
    else:
        raise NotImplemented("This dataset is not implemented yet : " + flagSettings.data_set)


def balanced_subsample(dataset, subsample_size=0.1):
    """
    Takes in a dataset and returns a subset of it given subsample_size. The class representation is forced to be the same
    :param dataset:
    :param subsample_size:
    :return:
    """

    x = dataset[0]
    y = dataset[1]

    # Calculate the calculate subsample
    desired_amount = x.shape[0] * subsample_size
    desired_amount_per_class = round(desired_amount / flagSettings.num_classes)

    # Takes two list and shuffles them together in order
    def shuffle_data(input1, input2):
        zipped = list(zip(input1, input2))
        random.shuffle(zipped)
        tmp = zip(*zipped)
        return tmp

    classes = []
    # Store classes individually
    for i in np.unique(y):
        indices = np.where(np.any(y == i, axis=1))
        shuffled_x, shuffled_y = shuffle_data(x[indices], y[indices])  # Shuffle the data
        classes.append((shuffled_x, shuffled_y))

    subsample_data_x = []
    subsample_data_y = []
    for i in np.unique(y):
        random_class_x = classes[i][0][:desired_amount_per_class]  # We can take the n first as it is shuffled
        random_class_y = classes[i][1][:desired_amount_per_class]
        subsample_data_x.extend(random_class_x)
        subsample_data_y.extend(random_class_y)
    subsample_data_x = np.array(subsample_data_x)
    subsample_data_y = np.array(subsample_data_y)

    # Shuffle order, else it will be returned in class order
    subsample_data_x, subsample_data_y = shuffle_data(subsample_data_x, subsample_data_y)

    # To array format
    subsample_data_x = np.array(subsample_data_x)
    subsample_data_y = np.array(subsample_data_y)

    return subsample_data_x, subsample_data_y


def save_test_accuracy(test_accuracy, file_name):
    print("Saving test accuracy: " + str(test_accuracy) + " to file" + file_name)
    with open(file_name, 'w') as f:
        f.write("Test accuracy: " + str(test_accuracy))
    print("Done writing results to file")
