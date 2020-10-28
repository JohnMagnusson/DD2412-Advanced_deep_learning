"""
The projection head to be attached to the encoder network.
"""

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense


def add_projection_head(base_model, type_of_projection_head, is_resnet50=False):
    """
    Adds projection head layers to the base mode. However does not compile the model, only attach.
    :param base_model: The output of the hidden from the base model (ResNet)
    :param type_of_projection_head: Which type of head to be attached (linear, nonlinear and etc)
    :param is_resnet50: If the network is a ResNet-50 or not
    :return: The output of the base model and the attached projection head
    """

    if type_of_projection_head == "linear":
        return get_linear_head(base_model, is_resnet50=is_resnet50)
    elif type_of_projection_head == "nonlinear":
        return get_non_linear_head(base_model, is_resnet50=is_resnet50)
    elif type_of_projection_head == "none":
        return base_model  # We are just passing the input as outputs
    elif type_of_projection_head == "nonlinear_extended":
        return get_non_linear_extended_head(base_model, is_resnet50)
    elif type_of_projection_head == "nonlinear_swish":
        return get_non_linear_swish_head(base_model, is_resnet50)
    else:
        raise Exception("This mode for the projection head is not supported: " + str(type_of_projection_head))


def get_linear_head(base_model, is_resnet50=False):
    """
    Attach the linear projection head
    :param base_model: The output of the hidden from the base model (ResNet)
    :param is_resnet50: If the network is a ResNet-50 or not
    :return: Output of the last hidden layer in the MLP (projection head)
    """

    if is_resnet50:
        return Dense(512, name="projection_head_linear")(base_model)
    else:
        return Dense(32, name="projection_head_linear")(base_model)


def get_non_linear_head(base_model, is_resnet50=False):
    """
    Attach nonlinear projection head
    :param base_model: The output of the hidden from the base model (ResNet)
    :param is_resnet50: If the network is a ResNet-50 or not
    :return: Output of the last hidden layer in the MLP (projection head)
    """
    if is_resnet50:
        projection_1 = Dense(2048, name="projection_head_1")(base_model)
        projection_1 = Activation("relu")(projection_1)
        projection_2 = Dense(128, name="projection_head_2")(projection_1)
    else:
        projection_1 = Dense(512, name="projection_head_1")(base_model)
        projection_1 = Activation("relu")(projection_1)
        projection_2 = Dense(32, name="projection_head_2")(projection_1)
    return projection_2


def get_non_linear_extended_head(base_model, is_resnet50=False):
    """
    An extend version of the original nonlinear projection head
    :param base_model: The output of the hidden from the base model (ResNet)
    :param is_resnet50: If the network is a ResNet-50 or not
    :return: Output of the last hidden layer in the MLP (projection head)
    """
    if is_resnet50:
        projection_1 = Dense(2048, name="projection_head_1")(base_model)
        projection_1 = Activation("relu")(projection_1)
        projection_2 = Dense(1024, name="projection_head_2")(projection_1)
        projection_2 = Activation("relu")(projection_2)
        projection_3 = Dense(128, name="projection_head_3")(projection_2)
    else:
        projection_1 = Dense(512, name="projection_head_1")(base_model)
        projection_1 = Activation("relu")(projection_1)
        projection_2 = Dense(256, name="projection_head_2")(projection_1)
        projection_2 = Activation("relu")(projection_2)
        projection_3 = Dense(32, name="projection_head_3")(projection_2)
    return projection_3


def get_non_linear_swish_head(base_model, is_resnet50=False):
    """
    :param base_model: The output of the hidden from the base model (ResNet)
    :param is_resnet50: If the network is a ResNet-50 or not
    :return: Output of the last hidden layer in the MLP (projection head)
    """
    if is_resnet50:
        projection_1 = Dense(2048, name="projection_head_1")(base_model)
        projection_1 = Activation("swish")(projection_1)
        projection_2 = Dense(128, name="projection_head_2")(projection_1)
    else:
        projection_1 = Dense(512, name="projection_head_1")(base_model)
        projection_1 = Activation("swish")(projection_1)
        projection_2 = Dense(32, name="projection_head_2")(projection_1)
    return projection_2
