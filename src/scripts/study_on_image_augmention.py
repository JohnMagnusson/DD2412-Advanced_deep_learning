# This is a study in how augmentation affects the accuracy and how important they are.
import os

import flagSettings
from dataManagement import balanced_subsample, get_data_set
from modelFunctions import build_simCLR_model, warmup_model, train_model, fine_tune_model


def run_image_augmentation_study():
    list_of_augmentations = ["crop", "cutout", "color", "sobel", "gaussian_noise", "gaussian_blur", "rotate"]
    model, train_data, val_data = prepare_pipeline(dataset="cifar-10", test_name="test1", encoder_network="resnet-18")
    run_training_pipeline(model, train_data, val_data)


def run_training_pipeline(model, train_data, val_data):
    warmed_up_model, training_loss, validation_loss = warmup_model(model, train_data, val_data)
    warmed_up_model.save_weights("warmedup_models/simCLR_model_weights_3_warmedup")  # TODO change save path
    print("Done with warmup")

    print("Starting real training")
    trained_model, training_loss, validation_loss = train_model(model, train_data, val_data)
    trained_model.save_weights("saved_models/simCLR_model_weights_3")  # TODO change save path
    print("Done with training")

    print("Starting with fine-tuning")
    model.load_weights("saved_models/simCLR_model_weights_3")
    train_data_sub = balanced_subsample(train_data, flagSettings.percentage_fine_tune_data)
    validation_data_sub = balanced_subsample(val_data, flagSettings.percentage_fine_tune_data)

    fine_tuned_model, history_fine_tune = fine_tune_model(model, "nonlinear", train_data_sub, validation_data_sub)
    fine_tuned_model.save_weights("finetuned_models/simCLR_model_weights_3")  # TODO change save path
    print("Done with fine-tuning")


def prepare_pipeline(dataset="cifar-10", test_name="test", encoder_network="resnet-18",
                     projection_head_mode="nonlinear"):
    if os.path.exists(test_name):
        raise FileExistsError("There exits already a test with this name, delete or choose another name.")
    os.makedirs(test_name)

    model = build_simCLR_model(encoder_network="resnet-18", projection_head_mode="nonlinear")
    train_data, val_data, test_data = get_data_set(dataset)
    return model, train_data, val_data


if __name__ == "__main__":
    run_image_augmentation_study()
