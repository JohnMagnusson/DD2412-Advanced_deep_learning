# Functions to help the model work. Such as training etc
from keras import optimizers

from models import projectionHead
from models import resnet18
import flagSettings


def trainModel(model, trainParameters, trainingData, trainingLabels):
    return 0, []


def evaluateModel(trainedModel, testData, testLabels):
    return []


def buildModel(encoderNetwork="resnet-18", projectionHeadMode="linear"):

    if encoderNetwork == "resnet-18":
        baseModel = resnet18.resnet18(input_shape=flagSettings.input_shape, num_classes=flagSettings.num_classes)
        simCLR = projectionHead.addProjectionHead(baseModel, projectionHeadMode)
    elif encoderNetwork == "resnet-50":
        raise NotImplemented("Not yet implemented")
    else:
        raise Exception("Illegal type of encoder network: " + str(encoderNetwork))

    def contrastiveLoss():
        IdoStuffHereYes = 0
        return IdoStuffHereYes

    # Todo fix lars optimizer here and real values
    lars = optimizers.SGD(learning_rate=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-07)
    simCLR.compile(loss=contrastiveLoss, optimizer=lars, metrics=["accuracy"])
    return simCLR

def fineTuneModel(model):
    # Add dense with softmax for prediction
    return model
