import pickle

from modelFunctions import *  # If using GPU, make sure this module is imported first. Else you will get GPU init error
from dataManagement import *
from linearEvaluation import linear_evaluation_model

# Flags to trigger different parts of training
do_warmup_new_model = True
do_train_new_model = True
do_fine_tune_model = True
do_linear_evaluation = True

encoder_network = "resnet-18"
projection_head = "nonlinear"
model_name = "ResNet-18_default"

train_data, val_data, test_data = get_data_set(flagSettings.data_set)

if do_warmup_new_model:
    print("Starting warmup")
    model = build_simCLR_model(encoder_network=encoder_network, projection_head_mode=projection_head)
    trained_model, training_loss, validation_loss = warmup_model(model, train_data, val_data)
    trained_model.save_weights("warmedup_models/" + model_name)
    plot_loss(training_loss, validation_loss, should_save_figure=True, file_name="warmup_loss")
    print("Done with warmup")

if do_train_new_model:
    print("Starting pretraining")
    model = build_simCLR_model(encoder_network=encoder_network, projection_head_mode=projection_head)
    model.load_weights("warmedup_models/" + model_name)
    trained_model, training_loss, validation_loss = train_model(model, train_data, val_data)
    trained_model.save_weights("saved_models/" + model_name)
    plot_loss(training_loss, validation_loss, should_save_figure=True, file_name="training_loss")
    print("Done with pretraining")

if do_fine_tune_model:
    print("Starting with fine-tuning")
    model = build_simCLR_model(encoder_network=encoder_network, projection_head_mode=projection_head)
    model.load_weights("saved_models/" + model_name)

    train_data_sub = balanced_subsample(train_data, flagSettings.percentage_fine_tune_data)
    validation_data_sub = balanced_subsample(val_data, 0.5)

    fine_tuned_model, history_fine_tune = fine_tune_model(model, "nonlinear", train_data_sub, validation_data_sub)
    fine_tuned_model.save_weights("finetuned_models/" + model_name)
    plot_fine_tuning(history_fine_tune, should_save_figure=True, file_name="fine_tuning")
    test_acc = evaluate_model(fine_tuned_model, test_data)
    save_test_accuracy(test_acc, model_name)
    print("Done with fine-tuning")

if do_linear_evaluation:
    print("Starting with linear evaluation")
    model = build_simCLR_model(encoder_network=encoder_network, projection_head_mode=projection_head)
    model.load_weights("saved_models" + model_name)
    sk_learn_model, val_accuracy, test_acc = linear_evaluation_model(model, train_data, val_data, test_data,
                                                                     "nonlinear")
    pickle.dump(sk_learn_model, open("linear_evaluation/" + model_name, 'wb'))
    plot_linear_evaluation_accuracy(val_accuracy, should_save_figure=True, file_name="linear_evaluation")
    save_test_accuracy(test_acc, model_name)
    print("linear evaluation test accuracy: " + str(test_acc))
