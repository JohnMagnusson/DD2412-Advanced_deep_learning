# This file contain all the settings and flags that is changable for the model
import lossFunctions

# Data settings
data_set = "cifar-10"
num_classes = 10    # CIFAR-10

# Encoder settings
input_shape = (32, 32, 3)

# Training settings
learning_rate = 0.5
temperature = 0.1       # Temperature in the loss function
batch_size = 256
weight_decay = 10e-6
loss_function = lossFunctions.NT_Xent_loss
nr_epochs = 3

# Image augmentation settings
color_jitter_strength = 0.5


