import sys
sys.path.append("..")

from src.models.resnet18 import resnet18
from src.dataManagement import load_cifar10
from src.customTraining import TrainingEngine
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


validation_size = 5000
batch_size = 128
epochs = 220

if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_cifar10()

    print("Data loaded.")

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    input_shape = x_train.shape[1:]
    n_colors = x_train.shape[3]
    depth = n_colors * 6 + 2
    model = resnet18(input_shape, 10)



    training_module = TrainingEngine(model)
    training_module.fit(train_data,
                          test_data,
                          batch_size=batch_size,
                          epochs=epochs)


    scores = training_module.evaluate(test_data)
    print('Test loss:', scores[1])
    print('Test accuracy:', scores[0])

