from dataManagement import *
from model import *
from modelFunctions import trainModel, evaluateModel

print("Hello world")

trainingData, trainingLabels, testData, testLabels = getDataSet("cifar-10")
model = createModel()

trainParameters = "Some object storing the parameters nicely"

trainedModel, trainingStats = trainModel(model, trainParameters, trainingData, trainingLabels)
evaluationStats = evaluateModel(trainedModel, testData, testLabels)

# visualize(trainingStats)
# visualize(evaluationStats)
