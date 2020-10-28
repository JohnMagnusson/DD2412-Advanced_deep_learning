import math

import tensorflow as tf
from tensorflow.python.ops import math_ops

import flagSettings


class Learning_rate_scheduler:

    def get_learning_rate(self, epoch):
        raise Exception("Need to use an inherited class, not the abstract.")


class Linear_decay_lr_scheduler(Learning_rate_scheduler):
    """
    An linear learning rate.
    Start small and then ramps up.
    """

    def get_learning_rate(self, epoch):
        warmup_epochs = flagSettings.nr_epochs_warmup
        initial_lr = flagSettings.learning_rate
        return initial_lr * ((epoch + 1) / warmup_epochs)


class Cosine_decay_lr_scheduler(Learning_rate_scheduler):
    """
    Function made to work similar to tf.keras.experimental.CosineDecay.
    Decaying learning rate in cosine format.
    """

    def __init__(self, decay_steps, initial_learning_rate=0.01):
        self.decay_steps = decay_steps
        self.initial_learning_rate = initial_learning_rate

    def get_learning_rate(self, step):
        """
        Calculates the decaying learning rate based on steps.
        :param step: Current step in the training, can be an iteration or an epoch
        :return: learning rate in tensor.float32 format
        """

        completed_fraction = step / self.decay_steps
        cosine_decayed = 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
        return math_ops.cast(self.initial_learning_rate * cosine_decayed, tf.float32)
