import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

import src
from src import flagSettings
from src.models import projectionHead
from src.models import resnet18
import src.flagSettings

# Allows to run on GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def build_simCLR_model(encoder_network="resnet-18", projection_head_mode="linear"):
    if encoder_network == "resnet-18":
        inputs, base_model = resnet18.resnet18(input_shape=flagSettings.input_shape)
        outputs = projectionHead.add_projection_head(base_model, projection_head_mode)
        raise NotImplemented("Projection head is not working until we have created the loss function")
    elif encoder_network == "resnet-50":
        raise NotImplemented("Not yet implemented")
    else:
        raise Exception("Illegal type of encoder network: " + str(encoder_network))

    def contrastive_loss():
        ido_stuff_here_yes = 0
        return ido_stuff_here_yes

    # Todo fix lars optimizer here and set proper training paramters
    sim_clr = Model(inputs=inputs, outputs=outputs)
    lars = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
    sim_clr.compile(loss=contrastive_loss, optimizer=lars, metrics=["accuracy"])
    sim_clr.summary()
    return sim_clr


def build_normal_resnet(isorOwn= False):
    inputs, hiddens = resnet18.resnet18(input_shape=flagSettings.input_shape)
    outputs = Dense(flagSettings.num_classes, activation='softmax')(hiddens)
    model = Model(inputs=inputs, outputs=outputs)
    if isorOwn:
        SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
        model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD, metrics=["accuracy"])
    return model


def train_model(model, training_data, training_labels):
    model.fit(training_data, training_labels, epochs=flagSettings.nr_epochs, batch_size=flagSettings.batch_size)
    return model, []

'''
def NTXent_Loss(z_i, z_j, tau=1.0):
    #https://joaolage.com/notes-simclr-framework
    batch_size = tf.shape(z_i)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    
    logits_ii = tf.matmul(z_i, z_i, transpose_b=True) / tau 
    logits_jj = tf.matmul(z_j, z_j, transpose_b=True) / tau
    
    logits_ii = logits_ii - masks * np.inf
    logits_jj = logits_jj - masks * np.inf
    
    logits_ij = tf.matmul(z_i, z_j, transpose_b=True) / tau
    logits_ji = tf.matmul(z_j, z_i, transpose_b=True) / tau
    
    tf.nn.softmax_cross_entropy_with_logits(
    loss_i = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ij, logits_ii], 1))
    loss_j = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ji, logits_jj], 1))
    
    loss=tf.reduce_mean(loss_i+loss_j)
    
    return loss
'''

def evaluate_model(trainedModel, testData, testLabels):
    scores = trainedModel.evaluate(x=testData, y= testLabels, verbose=1)
    print("%s: %.2f%% on the test set" % (trainedModel.metrics_names[1], scores[1] * 100))

def fine_tune_model(model):
    # Add dense with softmax for prediction
    return model
