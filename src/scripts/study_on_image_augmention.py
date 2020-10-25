# This is a study in how augmentation affects the accuracy and how important they are.
import os
import pickle

from modelFunctions import *
from augmentationEngine import AugmentationStudy
from dataAugmentations import *
from dataManagement import get_data_set
from linearEvaluation import linear_evaluation_model

# The folder name where all the tests will be saved
folder_prefix = "augmentation_test/"
plot_prefix = "/plots/"


def run_image_augmentation_study():
    test_name = "server_run_10_25"
    encoder_network = "resnet-18"
    projection_head = "nonlinear"
    data_set = "cifar-10"

    test_accuracy_per_augment = []

    # augmentations = [crop_resize, cut_out, color_jitter, sobel, gaussian_noise, gaussian_blur, rotate_randomly]
    augmentations = [cut_out, crop_resize, color_jitter, sobel, gaussian_noise, gaussian_blur]

    # !!! Rotate randomly we do in a separate test as it needs special execution which slows down the program by 300%
    # augmentations = [rotate_randomly, cut_out, crop_resize, color_jitter, gaussian_noise, gaussian_blur, sobel]

    train_data, val_data, test_data = prepare_pipeline(dataset=data_set, test_name=test_name)

    for i in range(len(augmentations)):
        for j in range(i, len(augmentations)):  # Skip already done combos, (aug_1, aug_2) = (aug_2, aug_1)
            model = build_simCLR_model(encoder_network=encoder_network, projection_head_mode=projection_head)

            if augmentations[i] == augmentations[j]:  # If they augmentations are the same we only do one of them.
                augmentation = AugmentationStudy(augmentations[i], nothing)
            else:
                augmentation = AugmentationStudy(augmentations[i], augmentations[j])

            # In the case of an error in the test, we continue with the next on in line
            try:
                test_accuracy = run_training_pipeline(model, train_data, val_data, test_data, augmentation, test_name)
                augmentation_test_name = str(augmentation.augmentation1.__name__)+"_" + str(augmentation.augmentation2.__name__)
                test_accuracy_per_augment.append((augmentation_test_name, test_accuracy))
                # Saves after iteration in case of crash or stopping
                save_test_accuracy(test_accuracy_per_augment, test_name)
            except:
                augmentation_test_name = str(augmentation.augmentation1.__name__)+"_" + str(augmentation.augmentation2.__name__)
                print("Got an excpetion whent rying to run test: " + augmentation_test_name +
                      ", please retry this test. Will continue now with next test.")


def run_training_pipeline(model, train_data, val_data, test_data, augmentation_engine, test_name):
    model_name = augmentation_engine.augmentation1.__name__ + "_" + augmentation_engine.augmentation2.__name__
    plot_save_path = folder_prefix + test_name + plot_prefix
    weights_save_path = folder_prefix + test_name

    print("Warming up model: " + model_name)
    # We inject the test name here so we save the checkpoint models under a test folder. A bit hackish but works well
    model._name = test_name + "/" + model_name + "-warmup"
    warmed_up_model, warmup_training_loss, warmup_validation_loss = warmup_model(model, train_data, val_data,
                                                                                 augmentation_engine)
    plot_loss(warmup_training_loss, warmup_validation_loss, should_save_figure=True,
              file_name=(plot_save_path + "warmup/" + model_name))
    warmed_up_model.save_weights(weights_save_path + "/warmup_models/" + model_name)
    print("Done with warmup")

    print("Starting pretraining on model: " + model_name)
    model._name = test_name + "/" + model_name + "-pretraining"
    trained_model, pre_training_loss, pre_validation_loss = train_model(model, train_data, val_data,
                                                                        augmentation_engine)
    plot_loss(pre_training_loss, pre_validation_loss, should_save_figure=True,
              file_name=(plot_save_path + "trained/" + model_name))
    trained_model.save_weights(weights_save_path + "/trained_models/" + model_name)
    print("Done with pretraining")

    print("Starting with linear evaluation: " + model_name)
    model._name = test_name + "/" + model_name + "-linear"
    sk_learn_model, val_accuracy, test_acc = linear_evaluation_model(trained_model, train_data, val_data, test_data,
                                                                     "nonlinear")
    plot_linear_evaluation_accuracy(val_accuracy, should_save_figure=True,file_name=(plot_save_path + "linear/" + model_name))
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


def save_test_accuracy(test_accuracy_per_augment, test_name="test"):
    print("Writing augmentation test accuracies to file")
    file_path_name = folder_prefix + test_name + ".txt"
    if not os.path.exists(folder_prefix + test_name):
        print("The folder for the test does not exist, saving the test_accuracies to project path")
        file_path_name = test_name + ".txt"
    with open(file_path_name, 'w') as f:
        f.write("Top 1% curries for the study on data augmentation\n")
        f.write("In case of duplicate in the augmentations, the augmentation is only run once.\n")
        for i in range(len(test_accuracy_per_augment)):
            f.write(
                "Augmentation: " + str(test_accuracy_per_augment[i][0]) + ", accuracy: " + str(test_accuracy_per_augment[i][1]))
            f.write("\n")
    print("Done writing results to file")


if __name__ == "__main__":
    run_image_augmentation_study()
