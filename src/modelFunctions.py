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


def build_simCLR_model(encoder_network="resnet-18", projection_head_mode="linear", weight_decay=False):
    """
    Builds a SimCLR model based on the paramters
    :param encoder_network: The encoder network of the SimCLR
    :param projection_head_mode: The projection head to be attached to the encoder network
    :param weight_decay: bool flag if use weight decay or not
    :return: A SimCLR model
    """

    if encoder_network == "resnet-18":
        inputs, base_model = resnet18.resnet18(input_shape=flagSettings.input_shape, weight_decay=weight_decay)
        outputs = projectionHead.add_projection_head(base_model, projection_head_mode)
        sim_clr = Model(inputs=inputs, outputs=outputs)
    elif encoder_network == "resnet-50":
        base_model = ResNet50(input_shape=flagSettings.input_shape,
                              include_top=False,
                              weights=None,
                              pooling='avg',
                              cifar=True)

        inputs = base_model.input
        outputs = projectionHead.add_projection_head(base_model.layers[-1].output, projection_head_mode,
                                                     is_resnet50=True)
        sim_clr = Model(inputs=inputs, outputs=outputs, name="SimCLR")
    else:
        raise Exception("Illegal type of encoder network: " + str(encoder_network))

    return sim_clr


def build_normal_resnet(is_or_own=False):
    """
    Builds a normal ResNet for supervised learning
    :param is_or_own: bool flag if the models should be used with our default settings or not
    :return: a ResNet model for supervised learning
    """

    inputs, hiddens = resnet18.resnet18(input_shape=flagSettings.input_shape)
    outputs = Dense(flagSettings.num_classes, activation='softmax')(hiddens)
    model = Model(inputs=inputs, outputs=outputs)
    if is_or_own:
        SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
        model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD, metrics=["accuracy"])
    return model


def warmup_model(model, train_data, val_data, augmentation_engine=SimClrAugmentation()):
    """
    Warms up the model
    :param model: The model to be warmed up
    :param train_data: Training data to be used
    :param val_data: Validation data to be used
    :param augmentation_engine: How the images will be augmented during training
    :return: A warmed up model, training and validation loss during the training. (model, training loss, validation loss)
    """

    training_module = TrainingEngine(model)
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
    """
    Train the model with unsupervised training
    :param model: The model to be trained
    :param train_data: Training data to be used
    :param val_data: Validation data to be used
    :param augmentation_engine: How the images will be augmented during training
    :return: A trained model, training and validation loss during the training. (model, training loss, validation loss)
    """

    training_module = TrainingEngine(model)
    training_module.optimizer = MomentumLARS(weight_decay=flagSettings.weight_decay)
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
    """
    Fine tunes the model on input data set. Freezing the base model
    :param base_model: The base model, already trained
    :param type_of_head: The type of projection head that was used during training
    :param train_dataset: Training data to be used
    :param validation_dataset: Validation data to be used
    :return: Fine tuned model and a history object with loss and accuracy for the training
    """

    if type_of_head == "nonlinear":
        fine_tune_at = -4
    elif type_of_head == "linear":
        fine_tune_at = -2
    elif type_of_head == "none":
        fine_tune_at = -1
    else:
        raise Exception("This type of head is not supported: " + str(type_of_head))

    x = base_model.layers[fine_tune_at].output
    base_model.trainable = False    # Here we freeze the base model, only the attached dense layer will be trained
    outputs = Dense(flagSettings.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer=tf.keras.optimizers.SGD(learning_rate=flagSettings.fine_tune_lr,
                                                    momentum=flagSettings.fine_tune_momentum,
                                                    nesterov=True))

    data_generator = generator_wrapper(
        ImageDataGenerator(preprocessing_function=fine_tune_augment).flow(x=train_dataset[0], y=train_dataset[1],
                                                                          batch_size=flagSettings.fine_tune_batch_size,
                                                                          shuffle=True))

    steps = train_dataset[0].shape[0] / flagSettings.fine_tune_batch_size
    history_fine_tune = model.fit(data_generator,
                                  epochs=flagSettings.fine_tune_nr_epochs,
                                  steps_per_epoch=steps,
                                  validation_data=validation_dataset)
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


def plot_loss(training_loss, validation_loss, should_save_figure=True, file_name=""):
    """
    Plots and saves the input losses, if specified
    :param training_loss: List of training loss
    :param validation_loss: List of validation loss
    :param should_save_figure: bool flag if the plot should be saved to file
    :param file_name: The name of the file if the plot is saved
    :return:
    """

    plt.plot(training_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Error [NTXent Loss]')
    plt.legend()
    plt.grid(True)
    if should_save_figure:
        plt.savefig(file_name + ".png")
    plt.show()


def plot_linear_evaluation_accuracy(validation_accuracy, should_save_figure=True, file_name=""):
    """
    Plots the validation accuracy of the linear evaluation, also saves plot if specified
    :param validation_accuracy:
    :param should_save_figure: bool flag if the plot should be saved to file
    :param file_name: The name of the file if the plot is saved
    :return:
    """

    plt.plot(validation_accuracy, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    if should_save_figure:
        plt.savefig(file_name + ".png")
    plt.show()


def plot_fine_tuning(history, should_save_figure=False, file_name=""):
    """
    Plots the fine-tuning information and saves if specified
    :param history: Fine-tuning history object (contains training and validation for both loss and accuracy
    :param should_save_figure: bool flag if the plot should be saved to file
    :param file_name: The name of the file if the plot is saved
    :return:
    """

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    if should_save_figure:
        plt.savefig(file_name + "-accuracy.png")
    plt.show()
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    if should_save_figure:
        plt.savefig(file_name + "-loss.png")
    plt.show()
    plt.clf()


def evaluate_model(trained_model, test_data):
    """
    Evaluates the trained model on test data
    :param trained_model: The trained model
    :param test_data: Test dataset
    :return: The models accuracy on the dataset
    """

    testData, testLabels = test_data[0], test_data[1]
    scores = trained_model.evaluate(x=testData, y=testLabels, verbose=1)
    print("%s: %.2f%% on the test set" % (trained_model.metrics_names[1], scores[1] * 100))
    return scores


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

    x, y = (dataset[0][:nr_sample_to_visualize], dataset[1][:nr_sample_to_visualize])
    x_features = projection.predict(x)
    x_in_low_space = tsne.fit_transform(x_features)
    plot_data_representation(x_in_low_space, y)
