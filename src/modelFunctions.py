import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

import src
from src.customTraining import TrainingEngine
from src import flagSettings, lossFunctions
from src.models import projectionHead
from src.models import resnet18

from src.augmentationEngine import SimClrAugmentation

# Allows to run on GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
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


    # Todo fix lars optimizer here and set proper training paramters

    # lars = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
    # sim_clr.compile(loss=contrastive_loss, optimizer=lars, metrics=["accuracy"])
    # sim_clr.summary()
    return sim_clr


def build_normal_resnet(isorOwn= False):
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

def train_model(model, train_data, test_data):
    training_module = TrainingEngine(model)
    training_module.optimizer = tf.keras.optimizers.SGD()
    training_module.train_loss = lossFunctions.NTXent_Loss()
    training_module.data_augmentation_module = SimClrAugmentation()
    training_module.fit(train_data,
                        test_data,
                        batch_size=flagSettings.batch_size,
                        epochs=flagSettings.nr_epochs)

    scores = training_module.evaluate(test_data)
    print('Test loss:', scores[1])
    print('Test accuracy:', scores[0])

    return model





def evaluate_model(trainedModel, testData, testLabels):
    scores = trainedModel.evaluate(x=testData, y= testLabels, verbose=1)
    print("%s: %.2f%% on the test set" % (trainedModel.metrics_names[1], scores[1] * 100))

def fine_tune_model(model):
    # Add dense with softmax for prediction
    return model
