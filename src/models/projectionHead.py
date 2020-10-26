from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense


def add_projection_head(base_model, projectionHeadMode, is_resnet50=False):
    if projectionHeadMode == "linear":
        return get_linear_head(base_model, is_resnet50=is_resnet50)
    elif projectionHeadMode == "nonlinear":
        return get_non_linear_head(base_model, is_resnet50=is_resnet50)
    elif projectionHeadMode == "none":
        return base_model  # We are just passing the input hiddens as output
    elif projectionHeadMode == "nonlinear_extended":
        return get_non_linear_extended_head(base_model, is_resnet50)
    elif projectionHeadMode == "nonlinear_swish":
        return get_non_linear_swish_head(base_model, is_resnet50)
    else:
        raise Exception("This mode for the projection head is not supported: " + str(projectionHeadMode))


def get_linear_head(base_model, is_resnet50=False):
    if is_resnet50:
        return Dense(2048, name="projection_head_linear")(base_model)
    else:
        return Dense(32, name="projection_head_linear")(base_model)


def get_non_linear_head(base_model, is_resnet50=False):
    """
    :param base_model: The output of the hidden from the base model (ResNet)
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
    An extend version of the original non-linear projection head
    :param is_resnet50:
    :param base_model: The output of the hidden from the base model (ResNet)
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
