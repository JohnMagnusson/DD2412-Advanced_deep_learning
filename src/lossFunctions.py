import numpy as np
import tensorflow as tf

'''
def NTXent_Loss(z_i, z_j):
    tau=1.0
    LARGE_NUM = 1e9

    # https://joaolage.com/notes-simclr-framework
    batch_size = tf.shape(z_i)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_ii = tf.matmul(z_i, z_i, transpose_b=True) / tau
    logits_jj = tf.matmul(z_j, z_j, transpose_b=True) / tau

    logits_ii = logits_ii - masks * LARGE_NUM
    logits_jj = logits_jj - masks * LARGE_NUM

    logits_ij = tf.matmul(z_i, z_j, transpose_b=True) / tau
    logits_ji = tf.matmul(z_j, z_i, transpose_b=True) / tau

    loss_i = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ij, logits_ii], 1))
    loss_j = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ji, logits_jj], 1))

    loss = tf.reduce_mean(loss_i + loss_j)

    return loss
'''
LARGE_NUM = 1e9

def NTXent_Loss(x, v, temperature=1.0):

    batch_size = tf.shape(x)[0]
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)

    logits_x_x = tf.matmul(x, x, transpose_b=True) / temperature
    logits_x_x = logits_x_x - masks * LARGE_NUM

    logits_v_v = tf.matmul(v, v, transpose_b=True) / temperature
    logits_v_v = logits_v_v - masks * LARGE_NUM

    logits_x_v = tf.matmul(x, v, transpose_b=True) / temperature
    logits_v_x = tf.matmul(v, x, transpose_b=True) / temperature

    loss_x = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_x_v, logits_x_x], 1))
    loss_v = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_v_x, logits_v_v], 1))

    loss = tf.reduce_mean(loss_x + loss_v)

    return loss