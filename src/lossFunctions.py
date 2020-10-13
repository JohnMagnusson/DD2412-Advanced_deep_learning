import numpy as np
import tensorflow as tf


def NTXent_Loss(z_i, z_j):
    tau=1.0
    # https://joaolage.com/notes-simclr-framework
    batch_size = tf.shape(z_i)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_ii = tf.matmul(z_i, z_i, transpose_b=True) / tau
    logits_jj = tf.matmul(z_j, z_j, transpose_b=True) / tau

    logits_ii = logits_ii - masks * np.inf
    logits_jj = logits_jj - masks * np.inf

    logits_ij = tf.matmul(z_i, z_j, transpose_b=True) / tau
    logits_ji = tf.matmul(z_j, z_i, transpose_b=True) / tau

    loss_i = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ij, logits_ii], 1))
    loss_j = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ji, logits_jj], 1))

    loss = tf.reduce_mean(loss_i + loss_j)

    return loss
