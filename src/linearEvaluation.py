import tensorflow as tf
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import flagSettings
import numpy as np
from tensorflow.keras.models import Model
from augmentationEngine import LinearEvalAugmentation
from dataAugmentations import fine_tune_augment


def linear_evaluation_model(model, train_data, val_data, test_data, type_of_head, n_classes=10):
    """
    Given a model and type of head, remove the head from the model and add a logistic linear regression
    classifier on top of the model and then train it.
    :return: sklearn model, validation accuracy list, test accuracy on test_data
    """

    if type_of_head == "nonlinear" or type_of_head == "nonlinear_swish":
        fine_tune_at = -4
    elif type_of_head == "linear":
        fine_tune_at = -2
    elif type_of_head == "none":
        fine_tune_at = -1
    elif type_of_head == "nonlinear_extended":
        fine_tune_at = -6
    else:
        raise Exception("This type of head is not supported: " + str(type_of_head))

    x = model.layers[fine_tune_at].output
    model.trainable = False
    model = Model(inputs=model.input, outputs=x)
    train_data = tf.data.Dataset.from_tensor_slices((tf.cast(train_data[0], dtype=tf.float64),
                                                     tf.keras.utils.to_categorical(train_data[1],
                                                                                   flagSettings.num_classes)))

    X_val = model.predict(val_data[0])
    X_test = model.predict(test_data[0])
    scikit_model = SGDClassifier(loss="log") # Logistic linear regression classifier when set to log

    val_accuracy = []
    augmEngine = LinearEvalAugmentation()
    classes_expanded = np.arange(0, n_classes, 1)
    for epoch in tqdm(range(flagSettings.linear_evaluation_nr_epochs)):
        shuffled_training_data = train_data.shuffle(len(list(train_data)))
        train_data_augmented = augmEngine.transform(shuffled_training_data)
        for batch_numb, (batch_xs, batch_ys) in enumerate(train_data_augmented.batch(flagSettings.linear_evaluation_batch_size), 1):

            batch_x = model.predict(batch_xs)
            scikit_model.partial_fit(batch_x, np.argmax(batch_ys.numpy(), axis=1).reshape(-1), classes=classes_expanded)
        y_val_pred = scikit_model.predict(tf.reshape(X_val, (-1, 512)))
        val_acc = accuracy_score(val_data[1], y_val_pred)
        val_accuracy.append(val_acc)
        print("Epoch: " + str(epoch) + ", Validation accuracy: " + str(val_acc))

    y_test_pred = scikit_model.predict(tf.reshape(X_test, (-1, 512)))
    test_acc = accuracy_score(test_data[1], y_test_pred)
    print("Test accuracy: " + str(test_acc))

    return scikit_model, val_accuracy, test_acc


def generator_wrapper(generator):
    """
    Wrapper function is used so we can have a pre_process function that use multiple inputs compared to the
    standard implementation from keras that only take the image batch as input.
    :param generator: The generator that will be used to generate data
    """

    while True:
        x = generator.next()

        images = []
        labels = []
        for i in range(x[0].shape[0]):
            images.append(fine_tune_augment(x[0][i]))
            labels.append(x[1][i])
        yield np.array(images), np.array(labels).flatten()
