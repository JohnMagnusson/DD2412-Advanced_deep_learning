import tensorflow as tf
import numpy as np


LARGE_NUM = 1e9
import flagSettings
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops


def  NT_Xent_loss(zis, zjs,
                 hidden_norm=True,
                 temperature=1.0,
                 tpu_context=None,
                 weights=1.0):

    # Get (normalized) hidden1 and hidden2.

    temperature = flagSettings.temperature
    hidden = tf.cast(tf.concat((zis, zjs), 0), dtype=tf.float32)

    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]


    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

    loss_a = tf.compat.v1.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
    loss_b = tf.compat.v1.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
    loss = loss_a + loss_b

    return loss


def logistic_loss(zis,zjs,hidden_norm=True):
    hidden = tf.cast(tf.concat((zis, zjs), 0), dtype=tf.float32)

    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
        
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    zis = tf.math.l2_normalize(zis, axis=1)
    zjs = tf.math.l2_normalize(zjs, axis=1)
    batch_size = shape(hidden1)[0]
    criterion =  tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    loss = criterion(y_pred=hidden1, y_true=hidden2)

    return loss/batch_size
    
    
def triplet_hard_loss(zi, zj,hidden_norm=True):
    hidden = tf.cast(tf.concat((zi, zj), 0), dtype=tf.float32)
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]
    
    y1 = tf.range(1, batch_size+1, delta=1, dtype=tf.int32, name='range')
    y2 = tf.range(1, batch_size+1, delta=1, dtype=tf.int32, name='range')
    labels = tf.concat((y1,y2),0)

    criterion =  tfa.losses.TripletSemiHardLoss(reduction=tf.keras.losses.Reduction.SUM, margin=1.0)
    loss = criterion(y_pred=hidden, y_true=labels)
    return loss


def lifted_loss(zi, zj, hidden_norm=True):
    hidden = tf.cast(tf.concat((zi, zj), 0), dtype=tf.float32)
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]
    
    y1 = tf.range(1, batch_size+1, delta=1, dtype=tf.int32, name='range')
    y2 = tf.range(1, batch_size+1, delta=1, dtype=tf.int32, name='range')
    labels = tf.concat((y1,y2),0)

    criterion = tfa.losses.LiftedStructLoss(reduction=tf.keras.losses.Reduction.SUM, margin=1.0)
    loss = criterion(y_pred=hidden, y_true=labels)
    return loss

