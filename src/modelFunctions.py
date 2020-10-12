import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from models import projectionHead
from models import resnet18
import flagSettings

# Allows to run on GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def build_simCLR_model(encoder_network="resnet-18", projection_head_mode="linear"):
    if encoder_network == "resnet-18":
        inputs, base_model = resnet18.resnet18(input_shape=flagSettings.input_shape)
        outputs = projectionHead.add_projection_head(base_model, projection_head_mode)
        raise NotImplemented("Projection head is not working until we have created the loss function")
    elif encoder_network == "resnet-50":
        raise NotImplemented("Not yet implemented")
    else:
        raise Exception("Illegal type of encoder network: " + str(encoder_network))

    def contrastive_loss():
        ido_stuff_here_yes = 0
        return ido_stuff_here_yes

    # Todo fix lars optimizer here and set proper training paramters
    sim_clr = Model(inputs=inputs, outputs=outputs)
    lars = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
    sim_clr.compile(loss=contrastive_loss, optimizer=lars, metrics=["accuracy"])
    sim_clr.summary()
    return sim_clr


def build_normal_resnet():
    inputs, hiddens = resnet18.resnet18(input_shape=flagSettings.input_shape)
    outputs = Dense(flagSettings.num_classes, activation='softmax')(hiddens)
    model = Model(inputs=inputs, outputs=outputs)
    SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
    model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD, metrics=["accuracy"])
    return model


def train_model(model, training_data, training_labels):
    model.fit(training_data, training_labels, epochs=flagSettings.nr_epochs, batch_size=flagSettings.batch_size)
    return model, []


def evaluate_model(trainedModel, testData, testLabels):
    scores = trainedModel.evaluate(x=testData, y= testLabels, verbose=1)
    print("%s: %.2f%% on the test set" % (trainedModel.metrics_names[1], scores[1] * 100))

def fine_tune_model(model):
    # Add dense with softmax for prediction
    return model
