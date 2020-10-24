from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.metrics import accuracy_score
import flagSettings
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



from augmentationEngine import LinearEvalAugmentation
from dataAugmentations import fine_tune_augment


weights = tf.Variable(tf.random.normal(shape=(2048, 10), dtype=tf.float64))
biases  = tf.Variable(tf.random.normal(shape=(10,), dtype=tf.float64))

def logistic_regression(x):
    lr = tf.add(tf.matmul(x, weights), biases)
    return tf.nn.sigmoid(lr)
    # return lr


def cross_entropy(y_true, y_pred):
    #y_true = tf.one_hot(y_true, 10)
    m = y_true.shape[0]
    cost = tf.multiply((1 / m), (-tf.tensordot(tf.transpose(y_true), tf.log(y_pred)) - tf.tensordot((1 - y_true).T, tf.log(1 - y_pred))))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return cost #tf.reduce_mean(loss)

def accuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_true = tf.cast(y_true, dtype=tf.int32)
    preds = tf.cast(tf.argmax(y_pred, axis=1), dtype=tf.int32)
    preds = tf.equal(y_true, preds)
    return tf.reduce_mean(tf.cast(preds, dtype=tf.float32))

def grad(model, x, y, y_middle):
    with tf.GradientTape() as tape:
        y_pred = logistic_regression(tf.cast(y_middle, dtype=tf.float64))
        loss_val = cross_entropy(y, y_pred)
    return tape.gradient(loss_val, [weights, biases])
'''
n_batches = 10000
learning_rate = 0.01
batch_size = 128

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().shuffle(x_train.shape[0]).batch(batch_size)


for batch_numb, (batch_xs, batch_ys) in enumerate(dataset.take(n_batches), 1):
    gradients = grad(batch_xs, batch_ys)
    optimizer.apply_gradients(zip(gradients, [weights, biases]))

    y_pred = logistic_regression(batch_xs)
    loss = cross_entropy(batch_ys, y_pred)
    acc = accuracy(batch_ys, y_pred)
    print("Batch number: %i, loss: %f, accuracy: %f" % (batch_numb, loss, acc))
'''
def linear_evaluation_model(model, train_data, val_data, test_data, type_of_head):
    #x_train = tf.reshape(train_data[0], shape=(-1, 128))
    #x_val = tf.reshape(val_data[0], shape=(-1, 128))
    if type_of_head == "nonlinear":
        fine_tune_at = -4
    elif type_of_head == "linear":
        fine_tune_at = -2
    elif type_of_head == "none":
        fine_tune_at = -1
    else:
        raise Exception("This type of head is not supported: " + str(type_of_head))

    # inputs = Input(shape=flagSettings.input_shape)
    # output_model = base_model(inputs)

    x = model.layers[fine_tune_at].output
    model.trainable = False
    model = Model(inputs=model.input, outputs=x)

    train_data = tf.data.Dataset.from_tensor_slices((tf.cast(train_data[0], dtype=tf.float64),
                                                     tf.keras.utils.to_categorical(train_data[1],
                                                                                   flagSettings.num_classes)))
    validation_data = tf.data.Dataset.from_tensor_slices((tf.cast(val_data[0], dtype=tf.float64),
                                                          tf.keras.utils.to_categorical(val_data[1],
                                                                           flagSettings.num_classes)))


    X_val = model.predict(val_data[0])
    X_test = model.predict(test_data[0])
    scikit_model = SGDClassifier(loss="log")#LogisticRegressionCV(multi_class='multinomial',  max_iter=1000, verbose=1)
    #scikit_model.fit(tf.reshape(X_train, (-1,2048)), train_data[1])

    val_accuracy = []
    augmEngine = LinearEvalAugmentation()
    for epoch in tqdm(range(flagSettings.linear_evaluation_nr_epochs)):
        shuffled_training_data = train_data.shuffle(len(list(train_data)))
        train_data_augmented = augmEngine.transform(shuffled_training_data)
        for batch_numb, (batch_xs, batch_ys) in enumerate(train_data_augmented.batch(flagSettings.linear_evaluation_batch_size), 1):

            batch_x = model.predict(batch_xs)
            scikit_model.partial_fit(batch_x, np.argmax(batch_ys.numpy(), axis=1).reshape(-1), classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_val_pred = scikit_model.predict(tf.reshape(X_val, (-1, 512)))
        val_acc = accuracy_score(val_data[1], y_val_pred)
        val_accuracy.append(val_acc)
        print("Epoch: " + str(epoch) + ", Validation accuracy: " + str(val_acc))

    y_test_pred = scikit_model.predict(tf.reshape(X_test, (-1, 512)))
    test_acc = accuracy_score(test_data[1], y_test_pred)
    print("Test accuracy: " + str(test_acc))

    return scikit_model, val_accuracy, test_acc
    '''
    y_test_pred = scikit_model.predict(tf.reshape(X_val, (-1, 2048)))

    from sklearn.metrics import classification_report
    print(classification_report(val_data[1], y_test_pred, scikit_model.classes_))
    print(accuracy_score(val_data[1], y_test_pred))
    '''
    '''
    optimizer = tf.optimizers.SGD(flagSettings.linear_evaluation_lr)
    for epoch in tqdm(range(flagSettings.linear_evaluation_nr_epochs)):
        for batch_numb, (batch_xs, batch_ys) in enumerate(train_data.batch(flagSettings.linear_evaluation_batch_size), 1):
            y_middle = model(batch_xs)
            gradients = grad(model, batch_xs, batch_ys, y_middle)
            optimizer.apply_gradients(zip(gradients, [weights, biases]))

            y_pred = logistic_regression(tf.cast(y_middle, dtype=tf.float64))
            loss = cross_entropy(batch_ys, y_pred)
            acc = accuracy(batch_ys, y_pred)
            print("Epoch: %i, Iteration: %i, loss: %f, accuracy: %f" % (epoch, batch_numb, loss, 
    '''

def le_test(base_model, type_of_head, train_dataset, validation_dataset, test_dataset):
    # Add dense with softmax for prediction
    if type_of_head == "nonlinear":
        fine_tune_at = -4
    elif type_of_head == "linear":
        fine_tune_at = -2
    elif type_of_head == "none":
        fine_tune_at = -1
    else:
        raise Exception("This type of head is not supported: " + str(type_of_head))

    x = base_model.layers[fine_tune_at].output
    base_model.trainable = False
    outputs = Dense(flagSettings.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer=tf.keras.optimizers.SGD(learning_rate=flagSettings.linear_evaluation_lr,
                                                    momentum=flagSettings.fine_tune_momentum,
                                                    nesterov=True))

    # model.summary()
    # data_generator = ImageDataGenerator(preprocessing_function=fine_tune_augment).flow(x=train_dataset[0],
    data_generator = generator_wrapper(
        ImageDataGenerator(preprocessing_function=fine_tune_augment).flow(x=train_dataset[0], y=train_dataset[1],
                                                                          batch_size=flagSettings.fine_tune_batch_size,
                                                                          shuffle=True))

    steps = train_dataset[0].shape[0] / flagSettings.fine_tune_batch_size
    history_fine_tune = model.fit(data_generator,
                                  epochs=flagSettings.fine_tune_nr_epochs,
                                  steps_per_epoch=steps,
                                  validation_data=validation_dataset)

    # history_fine_tune = model.fit(x=train_dataset[0], y=train_dataset[1],
    #                               epochs=flagSettings.fine_tune_nr_epochs,
    #                               validation_data=validation_dataset,
    #                               shuffle=True, batch_size=flagSettings.fine_tune_batch_size)

    return model, history_fine_tune

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
