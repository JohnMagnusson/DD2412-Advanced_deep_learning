# This is a study in how augmentation affects the accuracy and how important they are.
import os

import flagSettings
from augmentationEngine import AugmentationStudy
from dataManagement import balanced_subsample, get_data_set
from modelFunctions import build_simCLR_model, warmup_model, train_model, fine_tune_model

# The folder name where all the tests will be saved
folder_prefix = "augmentation_test/"


def run_image_augmentation_study():
    test_name = "test234"
    encoder_network = "resnet-18"
    projection_head = "nonlinear"
    data_set = "cifar-10"

    list_of_augmentations = ["cutout", "color", "sobel", "gaussian_noise", "gaussian_blur", "rotate"]
    train_data, val_data = prepare_pipeline(dataset=data_set, test_name=test_name)

    for aug_1 in list_of_augmentations:
        for aug_2 in list_of_augmentations:
            model = build_simCLR_model(encoder_network=encoder_network, projection_head_mode=projection_head)

            if aug_1 == aug_2:  # If they augmentations are the same we only do it once.
                augmentation = AugmentationStudy(aug_1, "nothing")
            else:
                augmentation = AugmentationStudy(aug_1, aug_2)
            run_training_pipeline(model, train_data, val_data, augmentation, test_name, projection_head)


def run_training_pipeline(model, train_data, val_data, augmentation_engine, test_name, projection_head):
    model_name = augmentation_engine.augmentation1 + "_" + augmentation_engine.augmentation2
    # We inject the test name here so we save the checkpoint models under a test folder. A bit hackish but works well
    model._name = test_name + "/" + model_name

    print("Warming up model: " + model_name)
    warmed_up_model, training_loss, validation_loss = warmup_model(model, train_data, val_data, augmentation_engine)
    warmed_up_model.save_weights(folder_prefix + test_name + "/warmedup_models/" + model_name)
    print("Done with warmup")

    print("Starting real training on model: " + model_name)
    trained_model, training_loss, validation_loss = train_model(model, train_data, val_data, augmentation_engine)
    trained_model.save_weights(folder_prefix + test_name + "/trained_models/" + model_name)
    print("Done with training")

    print("Starting with fine-tuning: " + model_name)
    train_data_sub = balanced_subsample(train_data, flagSettings.percentage_fine_tune_data)
    validation_data_sub = balanced_subsample(val_data, flagSettings.percentage_fine_tune_data)

    fine_tuned_model, history_fine_tune = fine_tune_model(model, projection_head, train_data_sub, validation_data_sub)
    fine_tuned_model.save_weights(folder_prefix + test_name + "/finetuned_models/" + model)
    print("Done with fine-tuning")


def prepare_pipeline(dataset="cifar-10", test_name="test"):
    # if os.path.exists(folder_prefix + test_name):
    #     raise FileExistsError("There exits already a test with this name, delete or choose another name.")
    # os.makedirs(folder_prefix + test_name)
    #
    # # Creates a folder for the checkpoint models
    # if os.path.exists("checkpoint_models/" + test_name):
    #     raise FileExistsError("There exits already a test with this name (in the checkpoint folder),"
    #                           " delete or choose another name.")
    # os.makedirs("../checkpoint_models/" + test_name)

    train_data, val_data, test_data = get_data_set(dataset)
    return train_data, val_data


if __name__ == "__main__":
    run_image_augmentation_study()
