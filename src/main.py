from dataManagement import *
from modelFunctions import *

do_warmup_new_model = False
do_train_new_model = False
do_fine_tune_model = False
do_evaluation_on_model = True

train_data, val_data, test_data = get_data_set(flagSettings.data_set)

if do_warmup_new_model:
    print("Starting warmup")
    model = build_simCLR_model(encoder_network="resnet-18", projection_head_mode="nonlinear")
    trained_model, training_loss, validation_loss = warmup_model(model, train_data, val_data)
    trained_model.save_weights("warmedup_models/simCLR_model_weights_different_loss_warmedup")
    plot_loss(training_loss, validation_loss, should_save_figure=True, file_name="warmup_loss")
    print("Done with warmup")

if do_train_new_model:
    print("Starting real training")
    model = build_simCLR_model(encoder_network="resnet-18", projection_head_mode="nonlinear")
    # model.load_weights("warmedup_models/simCLR_model_weights_different_loss_warmedup")
    trained_model, training_loss, validation_loss = train_model(model, train_data, val_data)
    trained_model.save_weights("saved_models/simCLR_model_weights_different_loss")
    plot_loss(training_loss, validation_loss, should_save_figure=True, file_name="training_loss")
    print("Done with training")

if do_fine_tune_model:
    print("Starting with fine-tuning")
    model = build_simCLR_model(encoder_network="resnet-18", projection_head_mode="nonlinear")
    # model.load_weights("saved_models/simCLR_model_weights_other_loss")
    model.load_weights("checkpoint_models/2020-10-22_09-41-37_1.5685284")
    train_data_sub = balanced_subsample(train_data, flagSettings.percentage_fine_tune_data)
    validation_data_sub = balanced_subsample(val_data, flagSettings.percentage_fine_tune_data)

    fine_tuned_model, history_fine_tune = fine_tune_model(model, "nonlinear", train_data_sub, validation_data_sub)
    fine_tuned_model.save_weights("finetuned_models/2020-10-22_09-41-37_1.5685284")
    plot_fine_tuning(history_fine_tune, should_save_figure=True, file_name="fine_tuning")
    print("Done with fine-tuning")

if do_evaluation_on_model:
    model = build_simCLR_model(encoder_network="resnet-18", projection_head_mode="nonlinear")
    model.load_weights("checkpoint_models/2020-10-22_09-41-37_1.5685284")
    visualize_model_class_understanding(model, test_data, nr_sample_to_visualize=1500, projection_head_mode="nonlinear")
