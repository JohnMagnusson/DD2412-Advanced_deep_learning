# This is a study in how augmentation affects the accuracy and how important they are.
import os

import flagSettings
from augmentationEngine import AugmentationStudy
from dataManagement import balanced_subsample, get_data_set
from modelFunctions import build_simCLR_model, warmup_model, train_model, fine_tune_model, evaluate_model, plot_loss

# The folder name where all the tests will be saved
folder_prefix = "augmentation_test/"
plot_prefix = "/plots/"


def run_image_augmentation_study():
    test_name = "test2"
    encoder_network = "resnet-18"
    projection_head = "nonlinear"
    data_set = "cifar-10"

    test_accuracy_per_augment = []

    # augmentations = ["cutout", "color", "sobel", "gaussian_noise", "gaussian_blur", "rotate", "crop"]
    augmentations = ["sobel", "color", "gaussian_noise", "gaussian_blur", "crop", "rotate", "cutout"]
    # augmentations = ["gaussian_noise", "gaussian_blur"]
    train_data, val_data, test_data = prepare_pipeline(dataset=data_set, test_name=test_name)

    for i in range(len(augmentations)):
        for j in range(i, len(augmentations)):  # Skip already done combos, (aug_1, aug_2) = (aug_2, aug_1)
            model = build_simCLR_model(encoder_network=encoder_network, projection_head_mode=projection_head)

            if augmentations[i] == augmentations[j]:  # If they augmentations are the same we only do one of them.
                augmentation = AugmentationStudy(augmentations[i], "nothing")
            else:
                augmentation = AugmentationStudy(augmentations[i], augmentations[j])
            test_accuracy = run_training_pipeline(model, train_data, val_data, test_data, augmentation, test_name, projection_head)
            test_accuracy_per_augment.append((augmentation, test_accuracy))

    save_test_accuracy(test_accuracy_per_augment, test_name)


def run_training_pipeline(model, train_data, val_data, test_data, augmentation_engine, test_name, projection_head):
    model_name = augmentation_engine.augmentation1 + "_" + augmentation_engine.augmentation2
    # We inject the test name here so we save the checkpoint models under a test folder. A bit hackish but works well
    model._name = test_name + "/" + model_name
    plot_save_path = folder_prefix + test_name + plot_prefix
    weights_save_path = folder_prefix + test_name

    print("Warming up model: " + model_name)
    warmed_up_model, warmup_training_loss, warmup_validation_loss = warmup_model(model, train_data, val_data, augmentation_engine)
    plot_loss(warmup_training_loss, warmup_validation_loss, should_save_figure=True, file_name=(plot_save_path + "warmup/" + model_name))
    warmed_up_model.save_weights(weights_save_path + "/warmup_models/" + model_name)
    print("Done with warmup")

    print("Starting pretraining on model: " + model_name)
    trained_model, pre_training_loss, pre_validation_loss = train_model(model, train_data, val_data,augmentation_engine)
    plot_loss(pre_training_loss, pre_validation_loss, should_save_figure=True, file_name=(plot_save_path + "trained/" + model_name))
    trained_model.save_weights(weights_save_path + "/trained_models/" + model_name)
    print("Done with pretraining")

    print("Starting with linear evaluation: " + model_name)
    # Todo replace with linear evaluation
    train_data_sub = balanced_subsample(train_data, flagSettings.percentage_fine_tune_data)
    validation_data_sub = balanced_subsample(val_data, flagSettings.percentage_fine_tune_data)
    fine_tuned_model, history_fine_tune = fine_tune_model(model, projection_head, train_data_sub, validation_data_sub)
    # fine_tuned_model, history_fine_tune = fine_tune_model(model, projection_head, train_data, val_data)
    fine_tuned_model.save_weights(weights_save_path + "/finetuned_models/" + model_name)

    test_accuracy = evaluate_model(fine_tuned_model, test_data)
    print("Done with linear evaluation")
    return test_accuracy


def prepare_pipeline(dataset="cifar-10", test_name="test"):
    # if os.path.exists(folder_prefix + test_name):
    #     raise FileExistsError("There exits already a test with this name, delete or choose another name.")
    # os.makedirs(folder_prefix + test_name)
    #
    # # # Creates a folder for the checkpoint models
    # if os.path.exists("../checkpoint_models/" + test_name):
    #     raise FileExistsError("There exits already a test with this name (in the checkpoint folder),"
    #                           " delete or choose another name.")
    # os.makedirs("../checkpoint_models/" + test_name)
    #
    # # Create folders for the models weights
    # os.makedirs(folder_prefix + test_name + "/warmup_models")
    # os.makedirs(folder_prefix + test_name + "/trained_models")
    # os.makedirs(folder_prefix + test_name + "/finetuned_models")
    #
    # # Plots folders
    # os.makedirs(folder_prefix + test_name + plot_prefix + "warmup")
    # os.makedirs(folder_prefix + test_name + plot_prefix + "trained")
    # os.makedirs(folder_prefix + test_name + plot_prefix + "finetuned")

    train_data, val_data, test_data = get_data_set(dataset)

    # TODO Tmp for testing
    nr_samples = 100
    # train_data = (train_data[0][:nr_samples],train_data[1][:nr_samples])
    # val_data = (val_data[0][:nr_samples],val_data[1][:nr_samples])
    test_data = (test_data[0][:nr_samples],test_data[1][:nr_samples])

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
                "Augmentation: " + test_accuracy_per_augment[i][0] + ", accuracy: " + test_accuracy_per_augment[i][1])
            f.write("\n")
    print("Done writing results to file")


if __name__ == "__main__":
    run_image_augmentation_study()
