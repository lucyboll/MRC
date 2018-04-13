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

def pointer(context_rep, hidden_state, cell, dtype=tf.float32):
    # context_rep: (B, T, 2*E)
    # hidden_state: (B, E)
    assert hidden_state.get_shape()[-1].value == cell.output_size
    size = context_rep.get_shape()[-1].value
    output_size = cell.output_size
    with tf.variable_scope('context_rep_linear_projection'):
        w = tf.get_variable('w_paras', [1, 1, size, output_size], dtype, tf.truncated_normal_initializer())
        context_rep_x = tf.expand_dims(context_rep, axis=2)
        feature = tf.nn.conv2d(context_rep_x, w, [1, 1, 1, 1], 'VALID')
        feature = tf.squeeze(feature, axis=2)
        v = tf.get_variable('v_paras', [output_size], dtype)
    with tf.variable_scope('hidden_state_linear_projection'):
        hidden_state_x = tf.layers.dense(hidden_state, units=output_size, use_bias=False)
        # hidden_state_x: (B, 1, E)
        hidden_state_x = tf.expand_dims(hidden_state_x, axis=1)
    with tf.variable_scope('contexts'):
        x = v * tf.tanh(hidden_state_x + feature)
        # score: (B, T)
        score = tf.reduce_sum(x, axis=-1)
        weight = tf.expand_dims(tf.nn.softmax(score), axis=1)
        context = tf.matmul(weight, context_rep)
        # context: (B, E)
        context = tf.squeeze(context, axis=1)
    with tf.variable_scope('rnn'):
        _, hidden_state = cell(context, hidden_state)
    return score, hidden_state

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

def gate(representation, context):
    assert representation.get_shape()[-1].value == context.get_shape()[-1].value
    size = representation.get_shape()[-1].value
    input = tf.concat([representation, context, representation * context], axis=-1)
    with tf.variable_scope('new_input'):
        new_input = tf.layers.dense(input, units=size, use_bias=False)
    with tf.variable_scope('gate'):
        gate = tf.layers.dense(input, units=size, activation=tf.sigmoid, use_bias=True)
    new_rep = gate * new_input
    return new_rep

def sfu(rep1, rep2):
    assert rep1.get_shape()[-1].value == rep2.get_shape()[-1].value
    size = rep1.get_shape()[-1].value
    input = tf.concat([rep1, rep2, rep1 * rep2], axis=-1)
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
            v * (tf.layers.dense(rep, units=size, use_bias=False) + tf.layers.dense(z, units=size, use_bias=False) +
                 tf.layers.dense(rep * z, units=size, use_bias=False)), axis=-1), axis=1)
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

def multihead_attention(query_antecedent, memory_antecedent, num_heads=8, mask=False):
    if memory_antecedent is None:
        memory_antecedent = query_antecedent
    size = query_antecedent.get_shape()[-1].value
    size1 = memory_antecedent.get_shape()[-1].value
    assert size == size1
    qs, ks, vs = [], [], []
    for i in xrange(num_heads):
        q = tf.layers.dense(query_antecedent, size, use_bias=False, name='q_%d' % i)
        k = tf.layers.dense(memory_antecedent, size, use_bias=False, name='k_%d' % i)
        v = tf.layers.dense(memory_antecedent, size, use_bias=False, name='v_%d' % i)
        qs.append(q)
        ks.append(k)
        vs.append(v)

    def scaled_dot_product_attention(q, k, v, mask, name):
        with tf.variable_scope(name):
            # q: [B, T1, E], k: [B, T2, E], v: [B, T2, E]
            # k: [B, T2, E] --> [B, E, T2]
            k = tf.transpose(k, [0, 2, 1])
            with tf.name_scope('MatMul_1'):
                # score: [B, T1, T2]
                score = tf.matmul(q, k)
            with tf.name_scope('Scale'):
                scaled_score = score / tf.sqrt(tf.to_float(size))
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
    output = tf.layers.dense(tf.concat(outputs, axis=-1), size, use_bias=False, name='Linear')
    return output

def ffn(inputs):
    filter_size = output_size = inputs.get_shape()[-1].value
    outputs = tf.layers.dense(inputs, filter_size, activation=tf.nn.relu, use_bias=True, name='ffn1')
    outputs = tf.layers.dense(outputs, output_size, use_bias=True, name='ffn2')
    return outputs

def conv_glu_v2(x, kernel_size, pad_length, output_size, batch_size, dtype=tf.float32):
    shape = x.get_shape()
    B = batch_size
    T = shape[1].value
    S = shape[-1].value
    pad_s = tf.zeros([B, pad_length, S], dtype=dtype, name='pad_s')
    pad_e = tf.zeros([B, pad_length, S], dtype=dtype, name='pad_e')
    x = tf.expand_dims(tf.concat([pad_s, x, pad_e], axis=1), axis=-1)
    kernel = tf.get_variable('kernel', [kernel_size, S, 1, 2*output_size], dtype=dtype)
    x = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    assert x.get_shape()[2].value == 1
    assert x.get_shape()[1].value == T
    x = tf.split(tf.squeeze(x, axis=2), 2, axis=-1)
    gate = tf.sigmoid(x[0])
    composition = tf.tanh(x[-1])
    x = gate * composition
    return x

def convolution(x, kernel_size, pad_length, output_size, batch_size, dtype=tf.float32):
    shape = x.get_shape()
    B = batch_size
    T = shape[1].value
    S = shape[-1].value
    pad_s = tf.zeros([B, pad_length, S], dtype=dtype, name='pad_s')
    pad_e = tf.zeros([B, pad_length, S], dtype=dtype, name='pad_e')
    x = tf.expand_dims(tf.concat([pad_s, x, pad_e], axis=1), axis=-1)
    kernel = tf.get_variable('kernel', [kernel_size, S, 1, output_size], dtype=dtype)
    x = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.relu(tf.squeeze(x, axis=2))
    return x

def short_cut(x, x_output, output_size):
    with tf.name_scope('short_cut'):
        size = x.get_shape()[-1].value
        x = tf.expand_dims(x, axis=2)
        kernel = tf.get_variable('kernel_short_cut', [1, 1, size, output_size], dtype=tf.float32)
        x = tf.nn.conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
        x = tf.squeeze(x, axis=2) + x_output
    with tf.variable_scope('layer_norm'):
        x = layer_norm(x)
    return x

def residual_drop_test(rep, rep_new, survival_rate):
    with tf.variable_scope('residual_drop_test'):
        size = rep_new.get_shape()[-1].value
        rep = short_cut(rep, survival_rate * rep_new, size)
    return rep

def residual_drop_train(rep, rep_new, gate):
    with tf.variable_scope('residual_drop_train'):
        size = rep_new.get_shape()[-1].value
        survival_rate = tf.cond(gate, lambda: 1.0, lambda: 0.0)
        rep = short_cut(rep, survival_rate * rep_new, size)
    return rep

def encoder_block(x, batch_size, sequence_length, dropout_rate, conv_layer_num, kernel_size, pad_length,
                  output_size, dropout):
    with tf.variable_scope('position_encoding'):
        size = x.get_shape()[-1].value
        p_encoding = position_encoding(sequence_length, size)
        x = x + p_encoding
    with tf.variable_scope('residual_block'):
        for i in xrange(conv_layer_num):
            with tf.variable_scope('convolution_%d' % i):
                x_ = conv_glu_v2(x, kernel_size, pad_length, output_size, batch_size)
            with tf.name_scope('short_cut_%d' % i):
                size = x.get_shape()[-1].value
                x = tf.expand_dims(x, axis=2)
                kernel = tf.get_variable('kernel%d' % i, [1, 1, size, output_size], dtype=tf.float32)
                x = tf.nn.conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
                x = tf.squeeze(x, axis=2) + x_
            with tf.variable_scope('layer_norm_%d' % i):
                x = layer_norm(x)
                x = tf.cond(dropout, lambda: tf.nn.dropout(x, keep_prob=1.0 - dropout_rate), lambda: x)
    with tf.variable_scope('attention_block'):
        with tf.variable_scope('attention_layer'):
            context = multihead_attention(x, x)
        with tf.variable_scope('short_cut'):
            size = x.get_shape()[-1].value
            x = tf.expand_dims(x, axis=2)
            kernel = tf.get_variable('kernel', [1, 1, size, output_size], dtype=tf.float32)
            x = tf.nn.conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
            x = tf.squeeze(x, axis=2) + context
        with tf.variable_scope('layer_norm'):
            x = layer_norm(x)
            x = tf.cond(dropout, lambda: tf.nn.dropout(x, keep_prob=1.0 - dropout_rate), lambda: x)
    with tf.variable_scope('ffn_block'):
        with tf.variable_scope('ffn_layer'):
            x_ffn = ffn(x)
        with tf.variable_scope('short_cut'):
            size = x.get_shape()[-1].value
            x = tf.expand_dims(x, axis=2)
            kernel = tf.get_variable('kernel', [1, 1, size, output_size], dtype=tf.float32)
            x = tf.nn.conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
            x = tf.squeeze(x, axis=2) + x_ffn
        with tf.variable_scope('layer_norm'):
            x = layer_norm(x)
            x = tf.cond(dropout, lambda: tf.nn.dropout(x, keep_prob=1.0 - dropout_rate), lambda: x)
    return x

def encoder_block_v1(x, batch_size, sequence_length, dropout_rate, conv_layer_num, kernel_size, pad_length,
                     output_size, dropout):
    with tf.variable_scope('position_encoding'):
        size = x.get_shape()[-1].value
        p_encoding = position_encoding(sequence_length, size)
        x = x + p_encoding
    with tf.variable_scope('residual_block'):
        for i in xrange(conv_layer_num):
            with tf.variable_scope('relu_block_%d' % i):
                with tf.variable_scope('convolution_layer1'):
                    x_ = convolution(x, kernel_size, pad_length, output_size, batch_size)
                with tf.variable_scope('convolution_layer2'):
                    x_ = convolution(x_, kernel_size, pad_length, output_size, batch_size)
                with tf.name_scope('short_cut'):
                    size = x.get_shape()[-1].value
                    x = tf.expand_dims(x, axis=2)
                    kernel = tf.get_variable('kernel%d' % i, [1, 1, size, output_size], dtype=tf.float32)
                    x = tf.nn.conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
                    x = tf.squeeze(x, axis=2) + x_
                with tf.variable_scope('layer_norm'):
                    x = layer_norm(x)
                    x = tf.cond(dropout, lambda: tf.nn.dropout(x, keep_prob=1.0 - dropout_rate), lambda: x)
            with tf.variable_scope('glu_block_%d' % i):
                with tf.variable_scope('convolution_layer'):
                    x_ = conv_glu_v2(x, 1, 0, output_size, batch_size)
                with tf.name_scope('short_cut'):
                    size = x.get_shape()[-1].value
                    x = tf.expand_dims(x, axis=2)
                    kernel = tf.get_variable('kernel', [1, 1, size, output_size], dtype=tf.float32)
                    x = tf.nn.conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
                    x = tf.squeeze(x, axis=2) + x_
                with tf.variable_scope('layer_norm'):
                    x = layer_norm(x)
                    x = tf.cond(dropout, lambda: tf.nn.dropout(x, keep_prob=1.0 - dropout_rate), lambda: x)
    with tf.variable_scope('attention_block'):
        with tf.variable_scope('attention_layer'):
            context = multihead_attention(x, x)
        with tf.variable_scope('short_cut'):
            size = x.get_shape()[-1].value
            x = tf.expand_dims(x, axis=2)
            kernel = tf.get_variable('kernel', [1, 1, size, output_size], dtype=tf.float32)
            x = tf.nn.conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
            x = tf.squeeze(x, axis=2) + context
        with tf.variable_scope('layer_norm'):
            x = layer_norm(x)
            x = tf.cond(dropout, lambda: tf.nn.dropout(x, keep_prob=1.0 - dropout_rate), lambda: x)
    with tf.variable_scope('ffn_block'):
        with tf.variable_scope('ffn_layer'):
            x_ffn = ffn(x)
        with tf.variable_scope('short_cut'):
            size = x.get_shape()[-1].value
            x = tf.expand_dims(x, axis=2)
            kernel = tf.get_variable('kernel', [1, 1, size, output_size], dtype=tf.float32)
            x = tf.nn.conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
            x = tf.squeeze(x, axis=2) + x_ffn
        with tf.variable_scope('layer_norm'):
            x = layer_norm(x)
            x = tf.cond(dropout, lambda: tf.nn.dropout(x, keep_prob=1.0 - dropout_rate), lambda: x)
    return x

def sru(x, output_size, dtype):

    def _create_ta(name, dtype):
        return tf.TensorArray(dtype, size=time_step, name=base_name + name)
    def call_cell(input_t, state, forget_t, reset_t):
        with tf.variable_scope('cell_state'):
            new_c = forget_t * state + (1 - forget_t) * input_t
        with tf.variable_scope('hidden_output'):
            new_h = reset_t * tf.tanh(new_c) + (1 - reset_t) * input_t
        return new_h, new_c
    def _time_step(time, output_ta_t, state):
        input_t = input_ta.read(time)
        forget_t = forget_ta.read(time)
        reset_t = reset_ta.read(time)
        output, new_state = call_cell(input_t, state, forget_t, reset_t)
        output_ta_t = output_ta_t.write(time, output)
        return time + 1, output_ta_t, new_state

    shape = [tf.shape(x)[0], output_size]
    time_step = tf.shape(x)[1]

    x = tf.transpose(x, perm=[1, 0, 2])

    time = tf.constant(0, tf.int32, name='time')
    cond = lambda time, *_: tf.less(time, time_step)
    with tf.name_scope('sru') as scope:
        base_name = scope
    output_ta = _create_ta('output', dtype)
    input_ta = _create_ta('input', dtype)
    forget_ta = _create_ta('forget', dtype)
    reset_ta = _create_ta('reset', dtype)
    with tf.variable_scope('x_hat'):
        x_hat = tf.layers.dense(x, units=output_size, use_bias=False)
        input_ta = input_ta.unstack(x_hat)
    with tf.variable_scope('gate'):
        gate = tf.layers.dense(x, units=2 * output_size, activation=tf.sigmoid, use_bias=True)
        r, f = tf.split(gate, num_or_size_splits=2, axis=-1)
        forget_ta = forget_ta.unstack(f)
        reset_ta = reset_ta.unstack(r)
    with tf.variable_scope('output'):
        state = tf.zeros(shape, dtype, name='zero_state')
        _, output_final_ta, final_state = tf.while_loop(
            cond=cond,
            body=_time_step,
            loop_vars=(time, output_ta, state)
        )
    final_output = output_final_ta.stack()
    final_output = tf.transpose(final_output, perm=[1, 0, 2])
    return final_output, final_state

def bi_sru(x, output_size, sequence_length=None, dtype=tf.float32):

    def _reverse(x, seq_lengths, seq_axis, batch_axis):
        if sequence_length is not None:
            x_reverse = tf.reverse_sequence(x, seq_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
        else:
            x_reverse = tf.reverse(x, axis=[1])
        return x_reverse

    with tf.variable_scope('fw'):
        output_fw, state_fw = sru(x, output_size, dtype)
    with tf.variable_scope('bw'):
        x_reverse = _reverse(x, sequence_length, 1, 0)
        output_bw, state_bw = sru(x_reverse, output_size, dtype)
        output_bw = _reverse(output_bw, sequence_length, 1, 0)
    outputs = (output_fw, output_bw)
    states = (state_fw, state_bw)
    return outputs, states

if __name__ == '__main__':
    x = tf.get_variable('x', [5, 10, 8], dtype=tf.float32)
    y = tf.get_variable('y', [5, 20, 8], dtype=tf.float32)
    x_seq_length = np.asarray([10, 5, 3, 8, 9], dtype=np.int32)
    y_seq_length = np.asarray([10, 5, 13, 8, 19], dtype=np.int32)
    with tf.variable_scope('x_encoding'):
        x_output, state = bi_sru(
            x=x,
            output_size=50,
            sequence_length=x_seq_length,
            dtype=tf.float32
        )
    with tf.variable_scope('y_encoding'):
        y_output, state = bi_sru(
            x=y,
            output_size=50,
            sequence_length=y_seq_length,
            dtype=tf.float32
        )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_out, y_out = sess.run([x_output, y_output])
        print(x_out, y_out)