# In this file we experiment how the loss/ accuracy changes as we choose different architectures for the projection head
import os

from dataManagement import get_data_set
from modelFunctions import build_simCLR_model

# The folder name where all the tests will be saved
folder_prefix = "augmentation_test/"
plot_prefix = "/plots/"


def run_projection_head_study():
    test_name = "test_pipeline"
    encoder_network = "resnet-18"
    projection_heads = ["nonlinear", "linear", "none"]  # Todo add new extension, convolutional?
    data_set = "cifar-10"

    train_data, val_data, test_data = prepare_pipeline(dataset=data_set, test_name=test_name)
    test_accuracy_for_projection_heads = []

    for projection_head in projection_heads:
        model = build_simCLR_model(encoder_network=encoder_network, projection_head_mode=projection_head)
        test_accuracy = run_training_pipeline(model, train_data, val_data, test_data, test_name, projection_head)
        test_accuracy_for_projection_heads.append((projection_head, test_accuracy))

    save_test_accuracy(test_accuracy_for_projection_heads, test_name)


def run_training_pipeline():
    # Todo continue from here
    pass

def prepare_pipeline(dataset="cifar-10", test_name="test"):
    train_data, val_data, test_data = get_data_set(dataset)
    return train_data, val_data, test_data


def save_test_accuracy(test_accuracy_for_projection_heads, test_name="test"):
    print("Writing test accuracies for different projection heads to file")
    file_path_name = folder_prefix + test_name + ".txt"
    if not os.path.exists(folder_prefix + test_name):
        print("The folder for the test does not exist, saving the test_accuracies to project path")
        file_path_name = test_name + ".txt"
    with open(file_path_name, 'w') as f:
        f.write("Top 1% curries for the study on projection head\n")
        for i in range(len(test_accuracy_for_projection_heads)):
            f.write(
                "Projection headd: " + test_accuracy_for_projection_heads[i][0] + ", accuracy: " +
                test_accuracy_for_projection_heads[i][1])
            f.write("\n")
    print("Done writing results to file")


if __name__ == "__main__":
    run_projection_head_study()
