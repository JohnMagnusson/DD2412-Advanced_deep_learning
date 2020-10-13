import sys
sys.path.append("..")

from src.models.resnet18 import resnet18
from src.dataManagement import load_cifar10
from src.customTraining import TrainingEngine
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from src.modelFunctions import build_normal_resnet
import flagSettings


validation_size = 5000
batch_size = 128
epochs = 220

if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_cifar10()

    print("Data loaded.")

    train_data = tf.data.Dataset.from_tensor_slices((tf.cast(x_train, dtype=tf.float32), tf.keras.utils.to_categorical(y_train, 10)))
    test_data = tf.data.Dataset.from_tensor_slices((tf.cast(x_test, dtype=tf.float32), tf.keras.utils.to_categorical(y_test, 10)))

    input_shape = x_train.shape[1:]
    n_colors = x_train.shape[3]
    depth = n_colors * 6 + 2
    model = resnet18(flagSettings.input_shape)

    '''
    model = tf.keras.applications.ResNet50(
      weights = None, input_shape = input_shape,
    pooling=None, classes=10)
    '''
    model = build_normal_resnet(isorOwn=True)

    '''
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
                
    '''
    training_module = TrainingEngine(model)
    training_module.optimizer = tf.keras.optimizers.SGD()
    training_module.fit(train_data,
                          test_data,
                          batch_size=flagSettings.batch_size,
                          epochs=flagSettings.nr_epochs)

    '''

    history = model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=2,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_test, y_test),
    )
    '''
    scores = training_module.evaluate(test_data)
    print('Test loss:', scores[1])
    print('Test accuracy:', scores[0])

