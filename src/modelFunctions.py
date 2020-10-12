import tensorflow as tf
from models import projectionHead
from models import resnet18
import flagSettings

# Allows to run on GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def build_model(encoder_network="resnet-18", projection_head_mode="linear"):
    if encoder_network == "resnet-18":
        base_model = resnet18.resnet18(input_shape=flagSettings.input_shape, num_classes=flagSettings.num_classes)
        sim_clr = projectionHead.addProjectionHead(base_model, projection_head_mode)
    elif encoder_network == "resnet-50":
        raise NotImplemented("Not yet implemented")
    else:
        raise Exception("Illegal type of encoder network: " + str(encoder_network))

    def contrastive_loss():
        ido_stuff_here_yes = 0
        return ido_stuff_here_yes

    # Todo fix lars optimizer here and real values
    lars = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
    sim_clr.compile(loss=contrastive_loss, optimizer=lars, metrics=["accuracy"])
    return sim_clr


def build_normal_resnet():
    baseModel = resnet18.resnet18(input_shape=flagSettings.input_shape, num_classes=flagSettings.num_classes)
    SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
    baseModel.compile(loss="sparse_categorical_crossentropy", optimizer=SGD, metrics=["accuracy"])
    return baseModel


def train_model(model, training_data, training_labels):
    model.fit(training_data, training_labels, epochs=flagSettings.nr_epochs, batch_size=flagSettings.batch_size)
    return model, []


def evaluate_model(trainedModel, testData, testLabels):
    scores = trainedModel.evaluate(x=testData, y= testLabels, verbose=1)
    print("%s: %.2f%% on the test set" % (trainedModel.metrics_names[1], scores[1] * 100))

def fine_tune_model(model):
    # Add dense with softmax for prediction
    return model
