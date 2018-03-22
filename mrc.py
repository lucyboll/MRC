#coding=utf-8
import tensorflow as tf
import numpy as np
from six.moves import xrange
from utils import attention
from utils import pointer
from utils import multiply_attention
from utils import sfu
from utils import fn
from utils import layer_norm
from utils import position_encoding
from utils import multihead_attention
from utils import ffn
class Model(object):
    def __init__(self, hps, word_emb, char_emb, dtype=tf.float32):
        self._hps = hps
        self.dtype = dtype
        self.word_emb = tf.constant(word_emb)
        self.char_emb = tf.constant(char_emb)
        self.global_step = tf.Variable(0, False)

        lr = tf.train.exponential_decay(hps.lr, self.global_step, 100, 0.99)
        min_lr = tf.Variable(0.0001, False)
        self.lr = tf.maximum(lr, min_lr)

        self.answer = [tf.placeholder(tf.int32, [None], 'answer{0}'.format(i)) for i in xrange(2)]
        self.context_word_inputs = tf.placeholder(tf.int32, [None, None], 'context_word_input')
        self.question_word_inputs = tf.placeholder(tf.int32, [None, None], 'question_word_input')
        self.context_lens = tf.placeholder(tf.int32, [None], 'context_lens')
        self.question_lens = tf.placeholder(tf.int32, [None], 'question_lens')

        self.context_char_inputs = tf.placeholder(tf.int32, [None, None], 'context_char_input')
        self.question_char_inputs = tf.placeholder(tf.int32, [None, None], 'question_char_input')

        context_word_embedding = tf.nn.embedding_lookup(self.word_emb, self.context_word_inputs)
        question_word_embedding = tf.nn.embedding_lookup(self.word_emb, self.question_word_inputs)
        context_char_embedding = tf.nn.embedding_lookup(self.char_emb, self.context_char_inputs)
        question_char_embedding = tf.nn.embedding_lookup(self.char_emb, self.question_char_inputs)

        with tf.variable_scope('char_representation'):
            char_cell = tf.contrib.rnn.GRUCell(hps.emb_size/2)
            _, state = tf.nn.bidirectional_dynamic_rnn(char_cell, char_cell, context_char_embedding, dtype=dtype)
            context_char_rep = tf.reshape(tf.concat(state, axis=-1), [hps.batch_size, hps.max_c_len, hps.emb_size])
            tf.get_variable_scope().reuse_variables()
            _, state = tf.nn.bidirectional_dynamic_rnn(char_cell, char_cell, question_char_embedding, dtype=dtype)
            question_char_rep = tf.reshape(tf.concat(state, axis=-1), [hps.batch_size, hps.max_q_len, hps.emb_size])

        self.context_inputs = tf.concat([context_word_embedding, context_char_rep], axis=-1)
        self.question_inputs = tf.concat([question_word_embedding, question_char_rep], axis=-1)

        self.build_graph()
        self.saver = tf.train.Saver(tf.global_variables())

    def r_net(self):
        hps = self._hps
        with tf.variable_scope('question_encoding'):
            q_rep = self.question_inputs
            q_states = []
            for i in xrange(hps.num_layers):
                with tf.variable_scope('layer%d' % i):
                    q_cell = tf.contrib.rnn.GRUCell(hps.size)
                    q_rep, q_state = tf.nn.bidirectional_dynamic_rnn(q_cell, q_cell, q_rep,
                                                                     sequence_length=self.question_lens,
                                                                     dtype=self.dtype)
                    q_rep = tf.concat(q_rep, axis=-1)
                    q_states.append(q_state)
            assert q_rep.get_shape()[-1].value == 2 * hps.size
        with tf.variable_scope('context_encoding'):
            c_rep = self.context_inputs
            for i in xrange(hps.num_layers):
                with tf.variable_scope('layer%d' % i):
                    c_cell = tf.contrib.rnn.GRUCell(hps.size)
                    c_rep, c_state = tf.nn.bidirectional_dynamic_rnn(c_cell, c_cell, c_rep,
                                                                     initial_state_fw=q_states[i][0],
                                                                     initial_state_bw=q_states[i][1],
                                                                     sequence_length=self.context_lens)
                    c_rep = tf.concat(c_rep, axis=-1)
            assert c_rep.get_shape()[-1].value == 2 * hps.size
        with tf.variable_scope('question_aware'):
            q_a_cell = tf.contrib.rnn.GRUCell(hps.size)
            context_q = multihead_attention(c_rep, q_rep)
            inputs = sfu(c_rep, context_q)
            c_rep, state = tf.nn.bidirectional_dynamic_rnn(q_a_cell, q_a_cell, inputs, self.context_lens,
                                                           dtype=self.dtype)
            c_rep = tf.concat(c_rep, axis=-1)
        with tf.variable_scope('self_attention'):
            s_a_cell = tf.contrib.rnn.GRUCell(hps.size)
            context_c = multihead_attention(c_rep, c_rep)
            inputs = sfu(c_rep, context_c)
            c_rep, state = tf.nn.bidirectional_dynamic_rnn(s_a_cell, s_a_cell, inputs, self.context_lens,
                                                           dtype=self.dtype)
            c_rep = tf.concat(c_rep, axis=-1)
            # if hps.mode == 'train':
            #     c_rep = tf.nn.dropout(c_rep, 1.0 - hps.dropout_rate)
            assert c_rep.get_shape()[-1].value == 2 * hps.size
        with tf.variable_scope('output_layer'):
            answer_cell = tf.contrib.rnn.GRUCell(2 * hps.size)
            with tf.variable_scope('pointer'):
                v_q = tf.get_variable('question_parameters', [hps.batch_size, 2 * hps.size], self.dtype,
                                      tf.truncated_normal_initializer())
                _, state = pointer(q_rep, v_q, answer_cell)
                tf.get_variable_scope().reuse_variables()
                start_pos_scores, state = pointer(c_rep, state, answer_cell)
                tf.get_variable_scope().reuse_variables()
                end_pos_scores, state = pointer(c_rep, state, answer_cell)
                self.pos_scores = [start_pos_scores, end_pos_scores]

    def mnemonic_reader(self):
        hps = self._hps
        with tf.variable_scope('question_encoding'):
            q_rep = self.question_inputs
            q_states = []
            for i in xrange(hps.num_layers):
                with tf.variable_scope('layer%d' % i):
                    q_cell = tf.contrib.rnn.GRUCell(hps.size)
                    q_rep, q_state = tf.nn.bidirectional_dynamic_rnn(q_cell, q_cell, q_rep,
                                                                     sequence_length=self.question_lens,
                                                                     dtype=self.dtype)
                    q_rep = tf.concat(q_rep, axis=-1)
                    q_states.append(q_state)
            assert q_rep.get_shape()[-1].value == 2 * hps.size
        with tf.variable_scope('context_encoding'):
            c_rep = self.context_inputs
            for i in xrange(hps.num_layers):
                with tf.variable_scope('layer%d' % i):
                    c_cell = tf.contrib.rnn.GRUCell(hps.size)
                    c_rep, c_state = tf.nn.bidirectional_dynamic_rnn(c_cell, c_cell, c_rep,
                                                                     initial_state_fw=q_states[i][0],
                                                                     initial_state_bw=q_states[i][1],
                                                                     sequence_length=self.context_lens)
                    c_rep = tf.concat(c_rep, axis=-1)
            assert c_rep.get_shape()[-1].value == 2 * hps.size
        with tf.variable_scope('iterative_aligner'):
            for i in xrange(hps.T):
                with tf.variable_scope('question_aware_%d' % i):
                    q_a_cell = tf.contrib.rnn.GRUCell(hps.size)
                    context_q = multihead_attention(c_rep, q_rep)
                    inputs = sfu(c_rep, context_q)
                    c_rep, state = tf.nn.bidirectional_dynamic_rnn(q_a_cell, q_a_cell, inputs, self.context_lens,
                                                                   dtype=self.dtype)
                    c_rep = tf.concat(c_rep, axis=-1)
                with tf.variable_scope('self_attention_%d' % i):
                    s_a_cell = tf.contrib.rnn.GRUCell(hps.size)
                    context_c = multihead_attention(c_rep, c_rep)
                    inputs = sfu(c_rep, context_c)
                    c_rep, state = tf.nn.bidirectional_dynamic_rnn(s_a_cell, s_a_cell, inputs, self.context_lens,
                                                                   dtype=self.dtype)
                    c_rep = tf.concat(c_rep, axis=-1)
                    # if hps.mode == 'train':
                    #     c_rep = tf.nn.dropout(c_rep, 1.0 - hps.dropout_rate)
                    assert c_rep.get_shape()[-1].value == 2 * hps.size
        with tf.variable_scope('memory_based_answer_pointer'):
            z_s = tf.expand_dims(tf.concat(q_state, axis=1), axis=1)
            for i in xrange(hps.L):
                with tf.variable_scope('start_position_%d' % i):
                    start_pos_scores, u_s = fn(c_rep, z_s)
                with tf.variable_scope('start_pos_memory_semantic_fusion_unit_%d' % i):
                    z_e = sfu(z_s, u_s)
                with tf.variable_scope('end_position_%d' % i):
                    end_pos_scores, u_e = fn(c_rep, z_e)
                with tf.variable_scope('end_pos_memory_semantic_fusion_unit_%d' % i):
                    z_s = sfu(z_e, u_e)
            self.pos_scores = [start_pos_scores, end_pos_scores]

    def transformer(self):
        hps = self._hps
        def transformer_encoder(encoder_input):
            x = encoder_input
            with tf.variable_scope('encoder'):
                with tf.variable_scope('self_attention'):
                    y = multihead_attention(x, None)
                    if hps.mode == 'train':
                        y = tf.nn.dropout(y, 1.0 - hps.dropout_rate)
                with tf.variable_scope('shortcut_norm_1'):
                    x += y
            return x

        with tf.variable_scope('question_encoding'):
            q_rep = self.question_inputs
            q_states = []
            for i in xrange(hps.num_layers):
                with tf.variable_scope('layer%d' % i):
                    q_cell = tf.contrib.rnn.GRUCell(hps.size)
                    q_rep, q_state = tf.nn.bidirectional_dynamic_rnn(q_cell, q_cell, q_rep,
                                                                     sequence_length=self.question_lens,
                                                                     dtype=self.dtype)
                    q_rep = tf.concat(q_rep, axis=-1)
                    q_states.append(q_state)
            assert q_rep.get_shape()[-1].value == 2 * hps.size
        with tf.variable_scope('context_encoding'):
            c_rep = self.context_inputs
            for i in xrange(hps.num_layers):
                with tf.variable_scope('layer%d' % i):
                    c_cell = tf.contrib.rnn.GRUCell(hps.size)
                    c_rep, c_state = tf.nn.bidirectional_dynamic_rnn(c_cell, c_cell, c_rep,
                                                                     initial_state_fw=q_states[i][0],
                                                                     initial_state_bw=q_states[i][1],
                                                                     sequence_length=self.context_lens)
                    c_rep = tf.concat(c_rep, axis=-1)
            assert c_rep.get_shape()[-1].value == 2 * hps.size
        with tf.variable_scope('iterative_aligner'):
            for i in xrange(hps.T):
                with tf.variable_scope('question_aware_%d' % i):
                    q_a_cell = tf.contrib.rnn.GRUCell(hps.size)
                    context_q = multihead_attention(c_rep, q_rep)
                    inputs = sfu(c_rep, context_q)
                    c_rep, state = tf.nn.bidirectional_dynamic_rnn(q_a_cell, q_a_cell, inputs, self.context_lens,
                                                                   dtype=self.dtype)
                    c_rep = tf.concat(c_rep, axis=-1)
                with tf.variable_scope('self_attention_%d' % i):
                    s_a_cell = tf.contrib.rnn.GRUCell(hps.size)
                    context_c = multihead_attention(c_rep, c_rep)
                    inputs = sfu(c_rep, context_c)
                    c_rep, state = tf.nn.bidirectional_dynamic_rnn(s_a_cell, s_a_cell, inputs, self.context_lens,
                                                                   dtype=self.dtype)
                    c_rep = tf.concat(c_rep, axis=-1)
                    # if hps.mode == 'train':
                    #     c_rep = tf.nn.dropout(c_rep, 1.0 - hps.dropout_rate)
                    assert c_rep.get_shape()[-1].value == 2 * hps.size
        with tf.variable_scope('residual_layer'):
            for i in xrange(6):
                with tf.variable_scope('layer_%d' % i):
                    c_rep = transformer_encoder(c_rep)
        with tf.variable_scope('output_layer'):
            answer_cell = tf.contrib.rnn.GRUCell(2 * hps.size)
            with tf.variable_scope('pointer'):
                v_q = tf.get_variable('question_parameters', [hps.batch_size, 2 * hps.size], self.dtype,
                                      tf.truncated_normal_initializer())
                _, state = pointer(q_rep, v_q, answer_cell)
                tf.get_variable_scope().reuse_variables()
                start_pos_scores, state = pointer(c_rep, state, answer_cell)
                tf.get_variable_scope().reuse_variables()
                end_pos_scores, state = pointer(c_rep, state, answer_cell)
                self.pos_scores = [start_pos_scores, end_pos_scores]

    def _loss(self):
        log_perplexity_list = []
        for score, ans in zip(self.pos_scores, self.answer):
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ans, logits=score)
            log_perplexity_list.append(crossent)
        log_perplexity = tf.add_n(log_perplexity_list)/2
        self.cost = tf.reduce_sum(log_perplexity)/self._hps.batch_size
        tf.summary.scalar('cost', self.cost)

    def train_op(self):
        tvars = tf.trainable_variables()
        with tf.device('/gpu:0'):
            grad, global_norm = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self._hps.max_grad_norm)
        tf.summary.scalar('global_norm', global_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        tf.summary.scalar('learning_rate', self.lr)
        self._train_op = optimizer.apply_gradients(zip(grad, tvars), global_step=self.global_step, name='train_step')

    def build_graph(self):
        if self._hps.model == 'r_net':
            self.r_net()
        elif self._hps.model == 'mnemonic_reader':
            self.mnemonic_reader()
        else:
            self.transformer()
        self._loss()
        if self._hps.mode == 'train':
            self.train_op()
        self._summary = tf.summary.merge_all()

    def run_trian_step(self, sess, context_words, question_words, context_chars, question_chars,
                       context_lens, question_lens, answer):
        input_feed = self.feed_data(context_words, question_words, context_chars, question_chars,
                       context_lens, question_lens, answer)
        output_feed = [self._train_op, self.cost, self._summary, self.global_step]
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs[1:]

    def run_eval_step(self, sess, context_words, question_words, context_chars, question_chars,
                       context_lens, question_lens, answer):
        input_feed = self.feed_data(context_words, question_words, context_chars, question_chars,
                                   context_lens, question_lens, answer)
        output_feed = [self.cost, self.pos_scores, self._summary, self.global_step]
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs

    def run_decode_step(self, sess, context_words, question_words, context_chars, question_chars,
                       context_lens, question_lens, answer):
        input_feed = self.feed_data(context_words, question_words, context_chars, question_chars,
                                   context_lens, question_lens, answer)
        output_feed = [self.pos_scores, self.global_step]
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs

    def feed_data(self, context_words, question_words, context_chars, question_chars,
                       context_lens, question_lens, answer):
        input_feed = {}
        input_feed[self.context_lens.name] = context_lens
        input_feed[self.question_lens.name] = question_lens
        input_feed[self.context_word_inputs.name] = context_words
        input_feed[self.question_word_inputs.name] = question_words
        input_feed[self.context_char_inputs.name] = context_chars
        input_feed[self.question_char_inputs.name] = question_chars
        for i in xrange(2):
            input_feed[self.answer[i].name] = answer[i]
        return input_feed