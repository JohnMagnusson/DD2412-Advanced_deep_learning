from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense


def add_projection_head(base_model, projectionHeadMode):
    if projectionHeadMode == "linear":
        return get_linear_head(base_model)
    elif projectionHeadMode == "nonlinear":
        return get_non_linear_head(base_model)
    elif projectionHeadMode == "none":
        return base_model  # We are just passing the input hiddens as output
    else:
        raise Exception("This mode for the projection head is not supported: " + str(projectionHeadMode))


def get_linear_head(base_model):
    return Dense(10, name="projection_head")(base_model)


def get_non_linear_head(base_model):
    """
    :param base_model: The output of the hidden from the base model (ResNet)
    :return: Output of the last hidden layer in the MLP (projection head)
    """
    projection_1 = Dense(10, name="projection_head_1")(base_model)
    projection_1 = Activation("relu")(projection_1)
    projection_2 = Dense(10, name="projection_head_2")(projection_1)
    projection_2 = Activation("relu")(projection_2)
    projection_3 = Dense(10, name="projection_head_3")(projection_2)
    return projection_3
