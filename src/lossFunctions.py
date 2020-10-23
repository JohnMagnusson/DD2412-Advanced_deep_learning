import tensorflow as tf

LARGE_NUM = 1e9
import flagSettings
import tensorflow as tf
import tensorflow_addons as tfa

def NTXent_Loss_test1(x, v, temperature=1.0):
    batch_size = tf.shape(x)[0]
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)

    logits_x_x = tf.matmul(x, x, transpose_b=True) / temperature
    logits_x_x = logits_x_x - masks * LARGE_NUM

    logits_v_v = tf.matmul(v, v, transpose_b=True) / temperature
    logits_v_v = logits_v_v - masks * LARGE_NUM

    logits_x_v = tf.matmul(x, v, transpose_b=True) / temperature
    logits_v_x = tf.matmul(v, x, transpose_b=True) / temperature

    loss_x = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_x_v, logits_x_x], 1))
    loss_v = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_v_x, logits_v_v], 1))

    loss = tf.reduce_mean(loss_x + loss_v)
    return loss


# From: https://stackoverflow.com/questions/62793043/tensorflow-implementation-of-nt-xent-contrastive-loss-function
# Define the contrastive loss function, NT_Xent (Tensorflow version)
#@tf.function
def NT_Xent_tf_test2(zi, zj, tau=1):
    """ Calculates the contrastive loss of the input data using NT_Xent. The
    equation can be found in the paper: https://arxiv.org/pdf/2002.05709.pdf
    (This is the Tensorflow implementation of the standard numpy version found
    in the NT_Xent function).

    Args:
        zi: One half of the input data, shape = (batch_size, feature_1, feature_2, ..., feature_N)
        zj: Other half of the input data, must have the same shape as zi
        tau: Temperature parameter (a constant), default = 1.

    Returns:
        loss: The complete NT_Xent constrastive loss
    """

    tau = flagSettings.temperature
    z = tf.cast(tf.concat((zi, zj), 0), dtype=tf.float32)
    loss = 0
    for k in range(zi.shape[0]):
        # Numerator (compare i,j & j,i)
        i = k
        j = k + zi.shape[0]
        # Instantiate the cosine similarity loss function
        cosine_sim = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        sim = tf.squeeze(- cosine_sim(tf.reshape(z[i], (1, -1)), tf.reshape(z[j], (1, -1))))
        numerator = tf.math.exp(sim / tau)

        # Denominator (compare i & j to all samples apart from themselves)
        sim_ik = - cosine_sim(tf.reshape(z[i], (1, -1)), z[tf.range(z.shape[0]) != i])
        sim_jk = - cosine_sim(tf.reshape(z[j], (1, -1)), z[tf.range(z.shape[0]) != j])
        denominator_ik = tf.reduce_sum(tf.math.exp(sim_ik / tau))
        denominator_jk = tf.reduce_sum(tf.math.exp(sim_jk / tau))

        # Calculate individual and combined losses
        loss_ij = - tf.math.log(numerator / denominator_ik)
        loss_ji = - tf.math.log(numerator / denominator_jk)
        loss += loss_ij + loss_ji

    # Divide by the total number of samples
    loss /= z.shape[0]

    return loss


def  NT_Xent_loss(zi, zj,
                 hidden_norm=True,
                 temperature=1.0,
                 tpu_context=None,
                 weights=1.0):
    """Compute loss for model.
    Args:
    hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    tpu_context: context information for tpu.
    weights: a weighting number or vector.
    Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.

    temperature = flagSettings.temperature
    hidden = tf.cast(tf.concat((zi, zj), 0), dtype=tf.float32)

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


def logistic_loss():
    return ""
    
def cosine_distance(features):
    distance = 1 - tf.matmul(features, features, transpose_b=True)
    return distance


def semi_hard_mining(distance, n_class_per_iter, n_img_per_class, threshold):
    N = n_class_per_iter * n_img_per_class
    N_pair = n_img_per_class * n_img_per_class

    # compute archor and pos indexes in numpy, reduce runtime compution
    pre_idx = np.arange(N, dtype='int64')
    arch_indexes = np.repeat(pre_idx, n_img_per_class, axis=0)
    pos_indexes = np.repeat(pre_idx.reshape(n_class_per_iter, n_img_per_class), n_img_per_class, axis=0).reshape(-1)
    pos_pair_indexes = np.stack([arch_indexes, pos_indexes], 1)

    # cast to tensorflow constant
    arch_indexes = tf.constant(arch_indexes)
    pos_indexes = tf.constant(pos_indexes)
    pos_pair_indexes = tf.constant(pos_pair_indexes)

    # gather distance
    with tf.control_dependencies([tf.assert_equal(tf.shape(distance)[0], N)]):
        pos_distance = tf.gather_nd(distance, pos_pair_indexes)
    neg_distance = tf.gather(distance, arch_indexes)

    # compute bool mask, false if it is pos index
    neg_pos_mask = np.ones(shape=[N*n_img_per_class, N], dtype='bool')
    for i in range(n_class_per_iter):
        neg_pos_mask[i*N_pair:(i+1)*N_pair, i*n_img_per_class:(i+1)*n_img_per_class] = 0
    neg_pos_mask = tf.constant(neg_pos_mask)

    # true if neg_dis - pos_dis < threshold
    candidate_mask = (neg_distance - tf.expand_dims(pos_distance, 1)) < threshold
    candidate_mask = tf.logical_and(candidate_mask, neg_pos_mask)

    # bool mask that delete triplets which has no candidate
    deletion_mask = tf.reduce_any(candidate_mask, axis=1)

    # deletion
    arch_indexes = tf.boolean_mask(arch_indexes, deletion_mask)
    pos_indexes = tf.boolean_mask(pos_indexes, deletion_mask)
    candidate_mask = tf.boolean_mask(candidate_mask, deletion_mask)

    # random sample candidation
    n_candidate_per_archer = tf.reduce_sum(tf.to_int32(candidate_mask), axis=1)
    sampler = tf.distributions.Uniform(0., tf.to_float(n_candidate_per_archer) - 1e-3)
    sample_idx = tf.to_int32(tf.floor(tf.reshape(sampler.sample(1), [-1])))
    start_idx = tf.cumsum(n_candidate_per_archer, exclusive=True)
    sample_idx = start_idx + sample_idx

    # collect neg_indexes
    candidate_indexes = tf.where(candidate_mask)
    neg_indexes = tf.gather(candidate_indexes, sample_idx)[:, 1]

    return (tf.stop_gradient(arch_indexes),
            tf.stop_gradient(pos_indexes),
            tf.stop_gradient(neg_indexes))


def triplet_hard_loss(pos_dis, neg_dis, threshold=0.2):
    return tf.reduce_mean(tf.nn.relu(threshold + pos_dis - neg_dis))
    
# def triplet_hard_loss(zi, zj):
#     return tfa.losses.TripletSemiHardLoss()