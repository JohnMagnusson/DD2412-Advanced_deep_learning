from dataManagement import *
from modelFunctions import train_model, evaluate_model, build_simCLR_model, build_normal_resnet, plot_loss



train_data, test_data = get_data_set()
model = build_simCLR_model(encoder_network="resnet-18", projection_head_mode="none")
# model = build_normal_resnet()

trained_model, trainingStats = train_model(model, train_data, test_data)
# evaluationStats = evaluate_model(trained_model, test_data, test_labels)
# plot_loss(trainingStats)


