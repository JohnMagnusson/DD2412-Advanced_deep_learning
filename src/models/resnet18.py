import tensorflow.keras
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, MaxPool2D
import tensorflow.keras.regularizers
from tensorflow.python.keras.regularizers import l2

import flagSettings


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 weight_decay=False):

    if weight_decay:
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_regularizer=l2(flagSettings.weight_decay_layers))
    else:
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same')

    x = inputs

    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)

    return x


def resnet18(input_shape, weight_decay=False):
    """ResNet-18

    -> Arguments
        input_shape (tensor): shape of input image tensor
        num_classes (int): number of classes

    -> Returns
        model (Model): Tensorflow model instance
    """

    num_filters = 64
    inputs = Input(shape=input_shape)

    if flagSettings.data_set == "cifar-10":
        x = resnet_layer(inputs=inputs, num_filters=num_filters, kernel_size=(3, 3), strides=1, weight_decay=weight_decay)
    else:
        x = resnet_layer(inputs=inputs, num_filters=num_filters, kernel_size=(7, 7), weight_decay=weight_decay)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)

    # Instantiate the stack of residual units
    num_res_blocks = [2, 2, 2, 2]
    for stack in range(4):
        for res_block in range(num_res_blocks[stack]):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             kernel_size=(3, 3),
                             num_filters=num_filters,
                             strides=strides,
                             weight_decay=weight_decay)
            y = resnet_layer(inputs=y,
                             kernel_size=(3, 3),
                             num_filters=num_filters,
                             activation=None,
                             weight_decay=weight_decay)
            if stack > 0 and res_block == 0:  # first layer but not first stack

                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=(1, 1),
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=True,
                                 weight_decay=weight_decay)
            x = tensorflow.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    x = GlobalAveragePooling2D()(x)
    return inputs, x  # We do not add the last layer here as it needs to be flexible for SimCLR or for a normal network
