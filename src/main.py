from dataManagement import *
from modelFunctions import train_model, evaluate_model, build_simCLR_model, build_normal_resnet


training_data, training_labels, test_data, test_labels = get_data_set()
model = build_simCLR_model(encoder_network="resnet-18", projection_head_mode="nonlinear")
# model = build_normal_resnet()

trained_model, trainingStats = train_model(model, training_data, training_labels)
evaluationStats = evaluate_model(trained_model, test_data, test_labels)

# visualize(trainingStats)
# visualize(evaluationStats)
