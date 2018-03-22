#coding=utf-8
import tensorflow as tf
import numpy as np
from six.moves import xrange
import math

def attention(rep1, rep2, cell, rep1_lens, dtype=tf.float32, num_heads=3):
    # rep1: (B, T1, E), rep2: (B, T2, E)
    if rep2 is None:
        rep2 = rep1
    size = rep2.get_shape()[-1].value
    # hidden_features: n * (B, T2, E)
    hidden_features = []
    v = []
    with tf.variable_scope('rep2_linear_projection'):
        for i in xrange(num_heads):
            w = tf.get_variable('w_paras_%d' % i, [1, 1, size, size], dtype, tf.random_uniform_initializer())
            rep2_x = tf.expand_dims(rep2, axis=2)
            feature = tf.nn.conv2d(rep2_x, w, [1, 1, 1, 1], 'VALID')
            feature = tf.squeeze(feature, axis=2)
            hidden_features.append(feature)
            v.append(tf.get_variable('v_paras_%d' % i, [size]))
    with tf.variable_scope('rep1_linear_projection'):
        w = tf.get_variable('w', [1, 1, size, size], dtype, tf.random_uniform_initializer())
        rep1_ = tf.expand_dims(rep1, axis=2)
        # rep1_x: (B, T1, 1, E)
        rep1_x = tf.nn.conv2d(rep1_, w, [1, 1, 1, 1], 'VALID')
        # rep1_x: (T1, B, 1, E)
        rep1_x = tf.transpose(rep1_x, perm=[1, 0, 2, 3])
    with tf.variable_scope('contexts'):
        contexts = []
        for i in xrange(num_heads):
            with tf.variable_scope('align_score_%d' % i):
                # x: (B, T1, T2, E)
                x = tf.transpose(hidden_features[i] + rep1_x, perm=[1, 0, 2, 3])
                x = v[i] * tf.tanh(x)
                # score: (B, T1, T2)
                score = tf.reduce_sum(x, axis=-1)
                weight = tf.nn.softmax(score)
                # context: (B, T1, E)
                context = tf.matmul(weight, rep2)
                contexts.append(context)
        context = tf.layers.dense(tf.concat(contexts, axis=-1), units=size, use_bias=True, name='contexts_linear_map')
    with tf.variable_scope('gated'):
        x = tf.concat([rep1, context], axis=-1)
        gate = tf.layers.dense(x, units=2*size, use_bias=True, activation=tf.nn.sigmoid, name='gate')
        gated_rep = gate * x
    with tf.variable_scope('birnn'):
        attentioned_rep, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, gated_rep, sequence_length=rep1_lens,
                                                                 dtype=dtype)
    attentioned_rep = tf.concat(attentioned_rep, axis=-1)
    state = tf.concat(state, axis=-1)
    return attentioned_rep, state

def pointer(context_rep, hidden_state, cell, dtype=tf.float32, num_heads=3):
    # context_rep: (B, T, 2*E)
    # hidden_state: (B, E)
    assert hidden_state.get_shape()[-1].value == cell.output_size
    size = context_rep.get_shape()[-1].value
    # hidden_features: n * (B, T, E)
    hidden_features = []
    v = []
    with tf.variable_scope('context_rep_linear_projection'):
        for i in xrange(num_heads):
            w = tf.get_variable('w_paras_%d' % i, [1, 1, size, size], dtype, tf.truncated_normal_initializer())
            context_rep_x = tf.expand_dims(context_rep, axis=2)
            feature = tf.nn.conv2d(context_rep_x, w, [1, 1, 1, 1], 'VALID')
            feature = tf.squeeze(feature, axis=2)
            hidden_features.append(feature)
            v.append(tf.get_variable('v_paras_%d' % i, [size], dtype))
    with tf.variable_scope('hidden_state_linear_projection'):
        hidden_state_x = tf.layers.dense(hidden_state, units=size, use_bias=True)
        # hidden_state_x: (B, 1, E)
        hidden_state_x = tf.expand_dims(hidden_state_x, axis=1)
    with tf.variable_scope('contexts'):
        contexts = []
        attention_scores = []
        for i in xrange(num_heads):
            x = tf.tanh(hidden_state_x + hidden_features[i])
            x = v[i] * x
            # score: (B, T)
            score = tf.reduce_sum(x, axis=-1)
            weight = tf.expand_dims(tf.nn.softmax(score), axis=1)
            context = tf.matmul(weight, context_rep)
            # context: (B, E)
            context = tf.squeeze(context, axis=1)
            contexts.append(context)
            attention_scores.append(score)
        context = tf.layers.dense(tf.concat(contexts, axis=-1), units=2 * size, use_bias=True, name='contexts_linear_map')
        scores = tf.add_n(attention_scores)
    with tf.variable_scope('rnn'):
        _, hidden_state = cell(context, hidden_state)
    return scores, hidden_state

def multiply_attention(rep1, rep2, time_step=None, diagonal_zero=False):
    # rep: (B, T, E)
    assert rep1.get_shape()[-1].value == rep2.get_shape()[-1].value
    # score: (B, T1, T2)
    score = tf.matmul(rep1, rep2, transpose_b=True)
    if diagonal_zero:
        diag = tf.ones([time_step])
        score = (1 - tf.matrix_diag(diag)) * score
    a = tf.nn.softmax(score)
    c = tf.matmul(a, rep2)
    return c

def sfu(rep1, rep2):
    assert rep1.get_shape()[-1].value == rep2.get_shape()[-1].value
    size = rep1.get_shape()[-1].value
    input = tf.concat([rep1, rep2, rep1 - rep2, rep1 * rep2], axis=-1)
    with tf.variable_scope('composition'):
        r = tf.layers.dense(input, units=size, use_bias=True, activation=tf.tanh)
    with tf.variable_scope('gate'):
        g = tf.layers.dense(input, units=size, use_bias=True, activation=tf.sigmoid)
    rep = g * r + (1-g) * rep1
    return rep

def fn(rep, z):
    # rep: (B, T, E)
    # z: (B, 1, E)
    size = z.get_shape()[-1].value
    v = tf.get_variable('V', [size], tf.float32)
    # score: (B, 1, T)
    score = tf.expand_dims(
        tf.reduce_sum(
            v * (tf.layers.dense(rep, units=size) + tf.layers.dense(z, units=size) +
                 tf.layers.dense(rep * z, units=size)), axis=-1), axis=1)
    # evidence vector
    u = tf.matmul(tf.nn.softmax(score), rep)
    return tf.squeeze(score, axis=1), u

def layer_norm(x, epsilon=1e-6):
    size = x.get_shape()[-1].value
    gain = tf.get_variable('gain', [size], dtype=tf.float32, initializer=tf.ones_initializer())
    bias = tf.get_variable('bias', [size], dtype=tf.float32, initializer=tf.zeros_initializer())

    mean = tf.reduce_mean(x, axis=-1, keep_dims=True)
    variance = tf.reduce_mean(tf.square(x-mean), axis=1, keep_dims=True)

    layer_norm_x = gain * tf.rsqrt(variance + epsilon) * (x - mean) + bias
    return layer_norm_x

def position_encoding(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal

def multihead_attention(query_antecedent, memory_antecedent, num_heads=4, mask=False):
    if memory_antecedent is None:
        memory_antecedent = query_antecedent
    size = query_antecedent.get_shape()[-1].value
    size1 = memory_antecedent.get_shape()[-1].value
    assert size == size1
    qs, ks, vs = [], [], []
    for i in xrange(num_heads):
        q = tf.layers.dense(query_antecedent, size, use_bias=True, name='q_%d' % i)
        k = tf.layers.dense(memory_antecedent, size, use_bias=True, name='k_%d' % i)
        v = tf.layers.dense(memory_antecedent, size, use_bias=True, name='v_%d' % i)
        qs.append(q)
        ks.append(k)
        vs.append(v)

    def scaled_dot_product_attention(q, k, v, mask, name):
        with tf.variable_scope(name):
            # q: [B, T1, E], k: [B, T2, E], v: [B, T2, E]
            t = k.get_shape()[1].value
            # k: [B, T2, E] --> [B, E, T2]
            k = tf.reshape(k, [-1, size, t])
            with tf.name_scope('MatMul_1'):
                # score: [B, T1, T2]
                score = tf.matmul(q, k)
            with tf.name_scope('Scale'):
                scaled_score = score / tf.sqrt(tf.to_float(size))
            if mask:
                with tf.name_scope('Mask'):
                    t_q = q.get_shape()[1].value
                    assert t_q == t
                    batch_size = q.get_shape()[0].value
                    score_mask = np.ones([batch_size, t, t], np.float32)
                    for i in xrange(batch_size):
                        for j in xrange(t):
                            for m in xrange(t):
                                if m > j:
                                    score_mask[i, j, m] = 0.0
                    scaled_score = scaled_score * score_mask
            with tf.name_scope('SoftMax'):
                weight = tf.nn.softmax(scaled_score)
            with tf.name_scope('MatMul_2'):
                # output: [B, T1, E]
                output = tf.matmul(weight, v)
        return output

    outputs = []
    for i in xrange(num_heads):
        output = scaled_dot_product_attention(qs[i], ks[i], vs[i], mask, name='head_%d' % i)
        outputs.append(output)
    output = tf.layers.dense(tf.concat(outputs, axis=-1), size, use_bias=True, name='Linear')
    return output

def ffn(inputs):
    filter_size = output_size = inputs.get_shape()[-1].value
    outputs = tf.layers.dense(inputs, filter_size, activation=tf.nn.relu, use_bias=True, name='ffn1')
    outputs = tf.layers.dense(outputs, output_size, use_bias=True, name='ffn2')
    return outputs

if __name__ == '__main__':
    a = tf.get_variable('a', [32, 10, 100], tf.float32, tf.zeros_initializer())
    b = tf.get_variable('b', [10, 32, 1, 100], tf.float32, tf.zeros_initializer())
    print((a + b).get_shape())