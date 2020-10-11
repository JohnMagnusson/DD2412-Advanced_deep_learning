import tensorflow as tf
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

import flagSettings


def linear_layer(input_shape, num_classes):
    layer = tf.keras.layers.Dense(input_shape=input_shape, units=num_classes, name="linear_layer")
    return layer


def addProjectionHead(baseModel, projectionHeadMode):
    if projectionHeadMode == "linear":
        return getLinearHead(baseModel)
    elif projectionHeadMode == "nonlinear":
        return getNonLinearHead(baseModel)
    elif projectionHeadMode == "none":
        return baseModel  # We are just passing the input hiddens as output
    else:
        raise Exception("This mode for the projection head is not supported: " + str(projectionHeadMode))


def getLinearHead(baseModel):
    # return Dense(256)(baseModel)
    return linear_layer(baseModel, flagSettings.proj_out_dim)


def getNonLinearHead(baseModel):
    projection_1 = Dense(256)(baseModel)
    projection_1 = Activation("relu")(projection_1)
    projection_2 = Dense(128)(projection_1)
    projection_2 = Activation("relu")(projection_2)
    projection_3 = Dense(50)(projection_2)
    inputs = Input(flagSettings.input_shape)
    return Model(inputs, projection_3)
