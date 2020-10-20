import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D

import flagSettings


def resnet50(input_shape):
    # Create resnet50- Check
    # Take input up until the stride change
    # Feed the splitted in to after maxpooling layer of resnet
    # Return model and input

    resnet_base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling='avg')
    resnet_base.summary()

    # inputs = Input(shape=input_shape)
    # model = Sequential()
    # model.add(inputs)
    # x = resnet_base.layers[0]
    # x = resnet_base.layers[1](x)
    #


    if flagSettings.data_set == "cifar-10": # For Cifar-10 we change the first conv to git the dataset
        x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(resnet_base.layers[1].output)
        top_half = Model(resnet_base.input, x)
        top_half.summary()

        middle_half = Model(resnet_base.layers[3].input, resnet_base.layers[5].output)
        middle_half.summary()
    # else:
    #     x = resnet_base.layers[2](x)

    # x.summary()
    #
    # for i in range(3, len(resnet_base.layers)):
    #     if i == 6 and flagSettings.data_set == "cifar-10":  # We do not add maxPooling in the beginning with Cifar-10
    #         pass
    #     else:
    #         model.add(resnet_base.layers[i])
    #     if i == 13:
    #         break

    # projection = Model(resnet_base.input, resnet_base.layers[-2].output)
    #
    # top_half = Model(resnet_base.input, baseModel.layers[5].output)
    #
    # projection.summary()

    test = 3

    return baseModel, ""
