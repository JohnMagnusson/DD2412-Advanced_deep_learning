from dataManagement import *
from model import *
from modelFunctions import trainModel, evaluateModel

print("Hello world")

trainDataSet, testDataSet = getDataSet("Ciphar-10")
model = createModel()

trainParameters = "Some object storing the parameters nicely"

trainedModel, trainingStats = trainModel(model, trainParameters, trainDataSet)
evaluationStats = evaluateModel(trainedModel, testDataSet)

# visualize(trainingStats)
# visualize(evaluationStats)
