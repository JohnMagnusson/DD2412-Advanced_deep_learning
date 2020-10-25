# In this file we experiment how the loss/ accuracy changes as we choose different architectures for the projection head
import os
import pickle

from dataManagement import get_data_set
from linearEvaluation import linear_evaluation_model
from modelFunctions import build_simCLR_model, warmup_model, train_model, plot_linear_evaluation_accuracy, plot_loss

# The folder name where all the tests will be saved
folder_prefix = "augmentation_test/"
plot_prefix = "/plots/"


def run_projection_head_study():
    test_name = "test_pipeline"
    encoder_network = "resnet-50"
    projection_heads = ["nonlinear_swish", "nonlinear_extended", "nonlinear", "linear", "none"]
    data_set = "cifar-10"

    train_data, val_data, test_data = prepare_pipeline(dataset=data_set, test_name=test_name)
    test_accuracy_for_projection_heads = []

    for projection_head in projection_heads:
        model = build_simCLR_model(encoder_network=encoder_network, projection_head_mode=projection_head)
        try:
            test_accuracy = run_training_pipeline(model, train_data, val_data, test_data, test_name, projection_head)
            test_accuracy_for_projection_heads.append((projection_head, test_accuracy))
            save_test_accuracy(test_accuracy_for_projection_heads, test_name)
        except:
            print("Got an exception when trying to run test: " + projection_head +
                  ", please retry this test. Will continue now with next test.")


def run_training_pipeline(model, train_data, val_data, test_data, test_name, projection_head):
    model_name = projection_head
    plot_save_path = folder_prefix + test_name + plot_prefix
    weights_save_path = folder_prefix + test_name

    print("Warming up model: " + model_name)
    # We inject the test name here so we save the checkpoint models under a test folder. A bit hackish but works well
    model._name = test_name + "/" + model_name + "-warmup"
    warmed_up_model, warmup_training_loss, warmup_validation_loss = warmup_model(model, train_data, val_data)
    plot_loss(warmup_training_loss, warmup_validation_loss, should_save_figure=True,
              file_name=(plot_save_path + "warmup/" + model_name))
    warmed_up_model.save_weights(weights_save_path + "/warmup_models/" + model_name)
    print("Done with warmup")

    print("Starting pretraining on model: " + model_name)
    model._name = test_name + "/" + model_name + "-pretraining"
    trained_model, pre_training_loss, pre_validation_loss = train_model(model, train_data, val_data)
    plot_loss(pre_training_loss, pre_validation_loss, should_save_figure=True,
              file_name=(plot_save_path + "trained/" + model_name))
    trained_model.save_weights(weights_save_path + "/trained_models/" + model_name)
    print("Done with pretraining")

    print("Starting with linear evaluation: " + model_name)
    model._name = test_name + "/" + model_name + "-linear"
    sk_learn_model, val_accuracy, test_acc = linear_evaluation_model(trained_model, train_data, val_data, test_data,
                                                                     "nonlinear")
    plot_linear_evaluation_accuracy(val_accuracy, should_save_figure=True,
                                    file_name=(plot_save_path + "linear/" + model_name))
    pickle.dump(sk_learn_model, open(weights_save_path + "/linear_models/" + model_name, 'wb'))
    print("Done with linear evaluation")
    return test_acc


def prepare_pipeline(dataset="cifar-10", test_name="test"):
    if os.path.exists(folder_prefix + test_name):
        raise FileExistsError("There exits already a test with this name, delete or choose another name.")
    os.makedirs(folder_prefix + test_name)

    # # Creates a folder for the checkpoint models
    if os.path.exists("../checkpoint_models/" + test_name):
        raise FileExistsError("There exits already a test with this name (in the checkpoint folder),"
                              " delete or choose another name.")
    os.makedirs("../checkpoint_models/" + test_name)

    # Create folders for the models weights
    os.makedirs(folder_prefix + test_name + "/warmup_models")
    os.makedirs(folder_prefix + test_name + "/trained_models")
    os.makedirs(folder_prefix + test_name + "/linear_models")

    # Plots folders
    os.makedirs(folder_prefix + test_name + plot_prefix + "warmup")
    os.makedirs(folder_prefix + test_name + plot_prefix + "trained")
    os.makedirs(folder_prefix + test_name + plot_prefix + "linear")

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
                "Projection head: " + str(test_accuracy_for_projection_heads[i][0]) + ", accuracy: " +
                str(test_accuracy_for_projection_heads[i][1]))
            f.write("\n")
    print("Done writing results to file")


if __name__ == "__main__":
    run_projection_head_study()