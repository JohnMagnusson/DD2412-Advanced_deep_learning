from typing import Optional

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

import flagSettings
from LARS_optimizer import MomentumLARS
from augmentationEngine import SimClrAugmentation
from customTraining import TrainingEngine
from models import projectionHead
from models import resnet18

# Allows to run on GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def build_simCLR_model(encoder_network="resnet-18", projection_head_mode="linear"):
    if encoder_network == "resnet-18":
        inputs, base_model = resnet18.resnet18(input_shape=flagSettings.input_shape)
        outputs = projectionHead.add_projection_head(base_model, projection_head_mode)
        sim_clr = Model(inputs=inputs, outputs=outputs)
    elif encoder_network == "resnet-50":
        raise NotImplemented("Not yet implemented")
    else:
        raise Exception("Illegal type of encoder network: " + str(encoder_network))

    sim_clr.summary()
    return sim_clr


def build_normal_resnet(isorOwn=False):
    inputs, hiddens = resnet18.resnet18(input_shape=flagSettings.input_shape)
    outputs = Dense(flagSettings.num_classes, activation='softmax')(hiddens)
    model = Model(inputs=inputs, outputs=outputs)
    if isorOwn:
        SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
        model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD, metrics=["accuracy"])
    return model


def train_model_default(model, training_data, training_labels):
    model.fit(training_data, training_labels, epochs=flagSettings.nr_epochs, batch_size=flagSettings.batch_size)
    return model, []


def train_model(model, train_data, val_data):
    training_module = TrainingEngine(model)

    # Todo make this compatible with warmup phase
    # Creating the cosine_decay learning rate schedule
    global_step = tf.compat.v1.train.get_or_create_global_step()
    nr_decay_steps = (train_data[0].shape[0] * flagSettings.nr_epochs) // flagSettings.batch_size
    lr_decay_fn = tf.compat.v1.train.cosine_decay(flagSettings.learning_rate, global_step, nr_decay_steps)

    training_module.optimizer = MomentumLARS(learning_rate=lr_decay_fn,
                                             weight_decay=flagSettings.weight_decay)
                                             # skip_list=['batch_normalization', 'bias',  'head_supervised'])   # Todo add this?

    training_module.loss_object = flagSettings.loss_function
    training_module.data_augmentation_module = SimClrAugmentation()
    training_loss, validation_loss = training_module.fit(train_data,
                                                         val_data,
                                                         batch_size=flagSettings.batch_size,
                                                         epochs=flagSettings.nr_epochs)

    return model, training_loss, validation_loss


def plot_loss(training_loss, validation_loss):
    plt.plot(training_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Error [NTXent Loss]')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fine_tuning(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def evaluate_model(trainedModel, testData, testLabels):
    scores = trainedModel.evaluate(x=testData, y=testLabels, verbose=1)
    print("%s: %.2f%% on the test set" % (trainedModel.metrics_names[1], scores[1] * 100))


def fine_tune_model(base_model, type_of_head, train_dataset, validation_dataset):
    # Add dense with softmax for prediction
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
    x = base_model.layers[fine_tune_at].output
    outputs = Dense(flagSettings.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer=tf.keras.optimizers.SGD(learning_rate=flagSettings.learning_rate,
                                                    momentum=flagSettings.fine_tune_momentum))

    model.summary()

    history_fine_tune = model.fit(x=train_dataset[0], y=train_dataset[1],
                                  epochs=flagSettings.fine_tune_nr_epochs,
                                  validation_data=validation_dataset,
                                  shuffle=True)

    return model, history_fine_tune
