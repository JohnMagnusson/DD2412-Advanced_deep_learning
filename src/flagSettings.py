# This file contain all the settings and flags that is changable for the model


# Data settings
data_set = "cifar-10"
num_classes = 10    # CIFAR-10

# Encoder settings
input_shape = (32, 32, 3)

# Training settings
nr_epochs = 2
batch_size = 64

# Projection head settings
proj_out_dim = 128  # Dimension on projection head output
