from dataManagement import *
from modelFunctions import trainModel, evaluateModel, buildModel


trainingData, trainingLabels, testData, testLabels = getDataSet()
model = buildModel(encoderNetwork="resnet-18", projectionHeadMode="linear")

trainParameters = "Some object storing the parameters nicely"

trainedModel, trainingStats = trainModel(model, trainParameters, trainingData, trainingLabels)
evaluationStats = evaluateModel(trainedModel, testData, testLabels)

# visualize(trainingStats)
# visualize(evaluationStats)
