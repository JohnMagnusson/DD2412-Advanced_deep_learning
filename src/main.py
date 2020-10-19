from dataManagement import *
from modelFunctions import *

do_warmup_new_model = True
do_train_new_model = False
do_fine_tune_model = False


train_data, val_data, test_data = get_data_set()
model = build_simCLR_model(encoder_network="resnet-18", projection_head_mode="nonlinear")

if do_warmup_new_model:
    trained_model, training_loss, validation_loss = warmup_model(model, train_data, val_data)
    plot_loss(training_loss, validation_loss)
    model.save_weights("simCLR_model_weights_2_warmedup")

if do_train_new_model:
    model.load_weights("saved_models/simCLR_model_weights_2_warmedup")
    trained_model, training_loss, validation_loss = train_model(model, train_data, val_data)
    plot_loss(training_loss, validation_loss)
    model.save_weights("simCLR_model_weights_2")

if do_fine_tune_model:
    model.load_weights("saved_models/simCLR_model_weights_2")
    train_data_sub = balanced_subsample(train_data, flagSettings.percentage_fine_tune_data)
    validation_data_sub = balanced_subsample(val_data, flagSettings.percentage_fine_tune_data)

    fine_tuned_model, history_fine_tune = fine_tune_model(model, "nonlinear", train_data_sub, validation_data_sub)
    plot_fine_tuning(history_fine_tune)
    fine_tuned_model.save_weights("saved_models/simCLR_model_fine_tuned")
# evaluationStats = evaluate_model(trained_model, test_data, test_labels)
