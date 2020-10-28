from modelFunctions import build_normal_resnet, generator_wrapper, evaluate_model, plot_fine_tuning
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import flagSettings
from dataManagement import get_data_set
from dataAugmentations import fine_tune_augment
import tensorflow as tf


def schedule(epoch):
    """
    Learning rate schedule function
    """

    lr = 0.05
    if epoch >= 60:
        return lr / 5
    elif epoch >= 120:
        return lr / (5 * 5)
    elif epoch >= 160:
        return lr / (5 * 5 * 5)
    else:
        return lr


if __name__ == "__main__":
    train_data, val_data, test_data = get_data_set()
    model = build_normal_resnet(isorOwn=False)
    data_generator = generator_wrapper(
        ImageDataGenerator(preprocessing_function=fine_tune_augment).flow(x=train_data[0], y=train_data[1],
                                                                          batch_size=flagSettings.fine_tune_batch_size,
                                                                          shuffle=True))

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

    SGD = tf.keras.optimizers.SGD(momentum=0.9)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD, metrics=["accuracy"])
    steps = train_data[0].shape[0] / flagSettings.fine_tune_batch_size
    history = model.fit(data_generator,
                        epochs=200,
                        validation_data=val_data,
                        batch_size=128,
                        steps_per_epoch=steps,
                        callbacks=[lr_schedule])

    plot_fine_tuning(history, should_save_figure=True, file_name="supervised_resnet18_cifar10")
    evaluate_model(model, test_data)
    model.save_weights("saved_models/supervised_resnet18_cifar10")
