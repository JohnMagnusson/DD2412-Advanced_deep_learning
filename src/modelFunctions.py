import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from LARS_optimizer import MomentumLARS
from augmentationEngine import SimClrAugmentation, fine_tune_augment
from customTraining import TrainingEngine
from learningRateSchedules import *
from models import projectionHead
from models import resnet18
from models.resnet50 import ResNet50

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
        base_model = ResNet50(input_shape=flagSettings.input_shape,
                              include_top=False,
                              weights=None,
                              pooling='avg',
                              cifar=True)
        inputs = base_model.input

        outputs = projectionHead.add_projection_head(base_model.layers[-1].output, projection_head_mode, resnet50=True)
        sim_clr = Model(inputs=inputs, outputs=outputs, name="SimCLR")
    else:
        raise Exception("Illegal type of encoder network: " + str(encoder_network))

    # sim_clr.summary()
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


def warmup_model(model, train_data, val_data, augmentation_engine=SimClrAugmentation()):
    training_module = TrainingEngine(model, set_custom_lr=True)

    training_module.optimizer = MomentumLARS()
    training_module.lr_scheduler = Linear_decay_lr_scheduler()
    training_module.loss_object = flagSettings.loss_function
    training_module.data_augmentation_module = augmentation_engine
    training_loss, validation_loss = training_module.fit(train_data,
                                                         val_data,
                                                         batch_size=flagSettings.batch_size,
                                                         epochs=flagSettings.nr_epochs_warmup)

    return model, training_loss, validation_loss


def train_model(model, train_data, val_data, augmentation_engine=SimClrAugmentation()):
    training_module = TrainingEngine(model, set_custom_lr=True)

    training_module.optimizer = MomentumLARS(weight_decay=flagSettings.weight_decay)
    # skip_list=['batch_normalization', 'bias',  'head_supervised'])   # Todo add this?

    training_module.lr_scheduler = Cosine_decay_lr_scheduler(decay_steps=flagSettings.nr_epochs,
                                                             initial_learning_rate=flagSettings.learning_rate)
    training_module.loss_object = flagSettings.loss_function
    training_module.data_augmentation_module = augmentation_engine
    training_loss, validation_loss = training_module.fit(train_data,
                                                         val_data,
                                                         batch_size=flagSettings.batch_size,
                                                         epochs=flagSettings.nr_epochs)

    return model, training_loss, validation_loss


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

    x = base_model.layers[fine_tune_at].output
    # base_model.trainable = False
    outputs = Dense(flagSettings.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer=tf.keras.optimizers.SGD(learning_rate=flagSettings.fine_tune_lr,
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


def plot_loss(training_loss, validation_loss, should_save_figure=False, file_name=""):
    plt.plot(training_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Error [NTXent Loss]')
    plt.legend()
    plt.grid(True)
    plt.show()
    if should_save_figure:
        plt.savefig(file_name + ".png")


def plot_fine_tuning(history, should_save_figure=False, file_name=""):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    if should_save_figure:
        plt.savefig(file_name + "-accuracy.png")

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    if should_save_figure:
        plt.savefig(file_name + "-loss.png")


def evaluate_model(trainedModel, testData, testLabels):
    scores = trainedModel.evaluate(x=testData, y=testLabels, verbose=1)
    print("%s: %.2f%% on the test set" % (trainedModel.metrics_names[1], scores[1] * 100))


def visualize_model_class_understanding(model, dataset, nr_sample_to_visualize, projection_head_mode="nonlinear"):
    """
    Visualizes how the model has learned the different classes in the data set
    :param model: The model that should visualize its idea of the data
    :param dataset: The dataset to visualize
    :param nr_sample_to_visualize: How many samples that shoudl be displayed
    :param projection_head_mode: If the model has any head, it will remove it
    :return: None
    """

    # Visualization of the representations
    def plot_data_representation(x_in_low_space, labels):
        fig = plt.figure()
        sns.set_style("darkgrid")
        labels = labels.reshape(-1)
        nr_colors = len(np.unique(labels))
        sns.scatterplot(x=x_in_low_space[:, 0], y=x_in_low_space[:, 1], hue=labels, legend='full',
                        palette=sns.color_palette("bright", nr_colors))
        plt.show()

    if projection_head_mode == "linear":
        projection = Model(model.input, model.layers[-2].output)
    elif projection_head_mode == "nonlinear":
        projection = Model(model.input, model.layers[-4].output)
    elif projection_head_mode == "none":
        projection = Model(model.input, model.layers[-1].output)
    else:
        raise Exception("This mode for the projection head is not supported: " + str(projection_head_mode))

    tsne = TSNE()
    projection.summary()

    x, y = (dataset[0][:nr_sample_to_visualize], dataset[1][:nr_sample_to_visualize])
    x_features = projection.predict(x)
    x_in_low_space = tsne.fit_transform(x_features)
    plot_data_representation(x_in_low_space, y)
