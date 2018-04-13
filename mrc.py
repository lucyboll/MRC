#coding=utf-8
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
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
from utils import short_cut
from utils import conv_glu_v2
from utils import encoder_block
from utils import encoder_block_v1
from utils import residual_drop_train
from utils import residual_drop_test
from utils import sru
from utils import bi_sru
from utils import gate

class Model(object):
    def __init__(self, hps, word_emb, char_emb, dtype=tf.float32):
        self._hps = hps
        self.dtype = dtype
        self.word_emb = tf.Variable(tf.constant(word_emb), trainable=False)
        self.char_emb = tf.Variable(tf.constant(char_emb), trainable=False)
        self.global_step = tf.Variable(0, False)

        lr = tf.train.exponential_decay(hps.lr, self.global_step, 1000, 0.99)
        min_lr = tf.Variable(0.0001, False)
        self.lr = tf.maximum(lr, min_lr)
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.max_c_length = tf.placeholder(tf.int32, [], name='max_c_length')
        self.max_q_length = tf.placeholder(tf.int32, [], name='max_q_length')
        self.max_char_length = tf.placeholder(tf.int32, [], name='max_char_length')
        self.dropout = tf.placeholder(tf.bool, [], name='dropout')
        self.answer = [tf.placeholder(tf.int32, [None], 'answer{0}'.format(i)) for i in xrange(2)]
        self.context_word_inputs = tf.placeholder(tf.int32, [None, None], 'context_word_input')
        self.question_word_inputs = tf.placeholder(tf.int32, [None, None], 'question_word_input')
        self.context_lens = tf.placeholder(tf.int32, [None], 'context_lens')
        self.question_lens = tf.placeholder(tf.int32, [None], 'question_lens')

        self.context_char_inputs = tf.placeholder(tf.int32, [None, None], 'context_char_input')
        self.question_char_inputs = tf.placeholder(tf.int32, [None, None], 'question_char_input')
        with tf.variable_scope('word_embedding'):
            context_word_embedding = tf.nn.embedding_lookup(self.word_emb, self.context_word_inputs)
            question_word_embedding = tf.nn.embedding_lookup(self.word_emb, self.question_word_inputs)
        with tf.variable_scope('char_embedding'):
            context_char_embedding = tf.nn.embedding_lookup(self.char_emb, self.context_char_inputs)
            question_char_embedding = tf.nn.embedding_lookup(self.char_emb, self.question_char_inputs)

        with tf.variable_scope('char_representation'):
            _, state = bi_sru(
                x=context_char_embedding,
                output_size=hps.char_dim // 2,
                dtype=self.dtype
            )
            context_char_rep = tf.reshape(tf.concat(state, axis=-1), [self.batch_size, self.max_c_length, hps.char_dim])
            tf.get_variable_scope().reuse_variables()
            _, state = bi_sru(
                x=question_char_embedding,
                output_size=hps.char_dim // 2,
                dtype=self.dtype
            )
            question_char_rep = tf.reshape(tf.concat(state, axis=-1), [self.batch_size, self.max_q_length, hps.char_dim])

        with tf.variable_scope('embedding'):
            context_inputs = tf.concat([context_word_embedding, context_char_rep], axis=-1)
            question_inputs = tf.concat([question_word_embedding, question_char_rep], axis=-1)
            self.context_inputs = tf.cond(self.dropout,
                                          lambda: tf.nn.dropout(context_inputs, keep_prob=1.0 - hps.dropout_rate),
                                          lambda: context_inputs)
            self.question_inputs = tf.cond(self.dropout,
                                           lambda: tf.nn.dropout(question_inputs, keep_prob=1.0 - hps.dropout_rate),
                                           lambda: question_inputs)

        self.build_graph()
        self.saver = tf.train.Saver(tf.global_variables())

    def r_net(self):
        hps = self._hps
        size = hps.size
        q_rep = self.question_inputs
        c_rep = self.context_inputs
        with tf.variable_scope('embedding_encoder_layer'):
            with tf.variable_scope('stacked_embedding_encoder_block'):
                # question encoding
                q_rep = encoder_block_v1(q_rep, self.batch_size, self.max_q_length, hps.dropout_rate, 4, 7, 3, size,
                                         self.dropout)
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('stacked_embedding_encoder_block'):
                # context encoding
                c_rep = encoder_block_v1(c_rep, self.batch_size, self.max_c_length, hps.dropout_rate, 4, 7, 3, size,
                                         self.dropout)
        with tf.variable_scope('context_question_attention_layer'):
            with tf.variable_scope('question_aware_context'):
                with tf.variable_scope('context'):
                    context_c = multihead_attention(q_rep, c_rep)
                with tf.variable_scope('question_semantic_fusion'):
                    q_rep = tf.concat([q_rep, context_c, q_rep * context_c], axis=-1)
                    q_rep = encoder_block_v1(q_rep, self.batch_size, self.max_q_length, hps.dropout_rate, 2, 7, 3,
                                             size, self.dropout)
            with tf.variable_scope('context_aware_question'):
                with tf.variable_scope('context'):
                    context_q = multihead_attention(c_rep, q_rep)
                with tf.variable_scope('context_semantic_fusion'):
                    c_rep = tf.concat([c_rep, context_q, c_rep * context_q], axis=-1)
                    for i in xrange(hps.num_stacks):
                        with tf.variable_scope('stack_%d' % i):
                            c_rep = encoder_block_v1(c_rep, self.batch_size, self.max_c_length,
                                                     hps.dropout_rate, 2, 7, 3, size, self.dropout)
                        # with tf.variable_scope('residual_drop_%d' % i):
                        #     death_rate = self.set_death_rate(i, hps.num_stacks, hps.last_rate)
                        #     rand = tf.random_uniform([], minval=0.0, maxval=1.0)
                        #     gate = tf.Variable(rand > death_rate, trainable=False)
                        #     c_rep = tf.cond(self.dropout,
                        #                     lambda: residual_drop_train(c_rep, c_rep_new, gate),
                        #                     lambda: residual_drop_test(c_rep, c_rep_new, 1.0 - death_rate))
        with tf.variable_scope('memory_based_answer_pointer'):
            with tf.variable_scope('init_state'):
                z_s = tf.reduce_mean(q_rep, axis=1, keep_dims=True)
                z_s = tf.cond(self.dropout, lambda: tf.nn.dropout(z_s, keep_prob=1.0 - hps.dropout_rate), lambda: z_s)
            for i in xrange(hps.T):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                with tf.variable_scope('start_position'):
                    start_pos_scores, u_s = fn(c_rep, z_s)
                with tf.variable_scope('start_pos_memory_semantic_fusion_unit'):
                    z_e = sfu(z_s, u_s)
                    z_e = tf.cond(self.dropout, lambda: tf.nn.dropout(z_e, keep_prob=1.0 - hps.dropout_rate),
                                  lambda: z_e)
                with tf.variable_scope('end_position'):
                    end_pos_scores, u_e = fn(c_rep, z_e)
                with tf.variable_scope('end_pos_memory_semantic_fusion_unit'):
                    z_s = sfu(z_e, u_e)
                    z_s = tf.cond(self.dropout, lambda: tf.nn.dropout(z_s, keep_prob=1.0 - hps.dropout_rate),
                                  lambda: z_s)
            self.pos_scores = [start_pos_scores, end_pos_scores]

    def mnemonic_reader(self):
        hps = self._hps
        size = hps.size
        q_rep = self.question_inputs
        c_rep = self.context_inputs
        with tf.variable_scope('embedding_encoder_layer'):
            with tf.variable_scope('stacked_embedding_encoder_block'):
                # question encoding
                q_rep = encoder_block(q_rep, self.batch_size, self.max_q_length, hps.dropout_rate, 4, 7, 3, size,
                                      self.dropout)
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('stacked_embedding_encoder_block'):
                # context encoding
                c_rep = encoder_block(c_rep, self.batch_size, self.max_c_length, hps.dropout_rate, 4, 7, 3, size,
                                      self.dropout)
        with tf.variable_scope('context_question_attention_layer'):
            with tf.variable_scope('question_aware_context'):
                with tf.variable_scope('context'):
                    context_c = multihead_attention(q_rep, c_rep)
                with tf.variable_scope('question_semantic_fusion'):
                    q_rep = tf.concat([q_rep, context_c, q_rep * context_c], axis=-1)
                    q_rep = encoder_block(q_rep, self.batch_size, self.max_q_length, hps.dropout_rate, 2, 7, 3,
                                          size, self.dropout)
            with tf.variable_scope('context_aware_question'):
                with tf.variable_scope('context'):
                    context_q = multihead_attention(c_rep, q_rep)
                with tf.variable_scope('context_semantic_fusion'):
                    c_rep = tf.concat([c_rep, context_q, c_rep * context_q], axis=-1)
                    for i in xrange(hps.num_stacks):
                        with tf.variable_scope('stack_%d' % i):
                            c_rep = encoder_block(c_rep, self.batch_size, self.max_c_length, hps.dropout_rate,
                                                  2, 7, 3, size, self.dropout)
                        # with tf.variable_scope('residual_drop_%d' % i):
                        #     death_rate = self.set_death_rate(i, hps.num_stacks, hps.last_rate)
                        #     rand = tf.random_uniform([], minval=0.0, maxval=1.0)
                        #     gate = tf.Variable(rand > death_rate, trainable=False)
                        #     c_rep = tf.cond(self.dropout,
                        #                     lambda: residual_drop_train(c_rep, c_rep_new, gate),
                        #                     lambda: residual_drop_test(c_rep, c_rep_new, 1.0 - death_rate))
        with tf.variable_scope('memory_based_answer_pointer'):
            with tf.variable_scope('init_state'):
                z_s = tf.reduce_mean(q_rep, axis=1, keep_dims=True)
                z_s = tf.cond(self.dropout, lambda: tf.nn.dropout(z_s, keep_prob=1.0 - hps.dropout_rate), lambda: z_s)
            for i in xrange(hps.T):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                with tf.variable_scope('start_position'):
                    start_pos_scores, u_s = fn(c_rep, z_s)
                with tf.variable_scope('start_pos_memory_semantic_fusion_unit'):
                    z_e = sfu(z_s, u_s)
                    z_e = tf.cond(self.dropout, lambda: tf.nn.dropout(z_e, keep_prob=1.0 - hps.dropout_rate),
                                  lambda: z_e)
                with tf.variable_scope('end_position'):
                    end_pos_scores, u_e = fn(c_rep, z_e)
                with tf.variable_scope('end_pos_memory_semantic_fusion_unit'):
                    z_s = sfu(z_e, u_e)
                    z_s = tf.cond(self.dropout, lambda: tf.nn.dropout(z_s, keep_prob=1.0 - hps.dropout_rate),
                                  lambda: z_s)
            self.pos_scores = [start_pos_scores, end_pos_scores]

    def transformer(self):

        hps = self._hps
        with tf.variable_scope('question_convolution_encoding'):
            q_rep = self.question_inputs
            q_output = conv_glu_v2(q_rep, 3, 1, hps.size, self.batch_size)
            q_output = tf.cond(self.dropout, lambda: tf.nn.dropout(q_output, keep_prob=1.0 - hps.dropout_rate),
                               lambda: q_output)
            q_rep = short_cut(q_rep, q_output, q_output.get_shape()[-1].value)
        with tf.variable_scope('context_convolution_encoding'):
            c_rep = self.context_inputs
            c_output = conv_glu_v2(c_rep, 3, 1, hps.size, self.batch_size)
            c_output = tf.cond(self.dropout, lambda: tf.nn.dropout(c_output, keep_prob=1.0 - hps.dropout_rate),
                               lambda: c_output)
            c_rep = short_cut(c_rep, c_output, c_output.get_shape()[-1].value)
        with tf.variable_scope('question_encoding'):
            for i in xrange(hps.num_layers):
                with tf.variable_scope('layer%d' % i):
                    q_rep, q_state = bi_sru(
                        x=q_rep,
                        output_size=hps.size,
                        sequence_length=self.question_lens,
                        dtype=self.dtype
                    )
                    q_rep = tf.concat(q_rep, axis=-1)
                    q_rep = tf.layers.dense(q_rep, units=hps.size, use_bias=False, name='q_rep')
                    q_rep = tf.cond(self.dropout, lambda: tf.nn.dropout(q_rep, keep_prob=1.0 - hps.dropout_rate),
                                       lambda: q_rep)
            assert q_rep.get_shape()[-1].value == hps.size
        with tf.variable_scope('context_encoding'):
            for i in xrange(hps.num_layers):
                with tf.variable_scope('layer%d' % i):
                    c_rep, c_state = bi_sru(
                        x=c_rep,
                        output_size=hps.size,
                        sequence_length=self.context_lens,
                        dtype=self.dtype
                    )
                    c_rep = tf.concat(c_rep, axis=-1)
                    c_rep = tf.layers.dense(c_rep, units=hps.size, use_bias=False, name='c_rep')
                    c_rep = tf.cond(self.dropout, lambda: tf.nn.dropout(c_rep, keep_prob=1.0 - hps.dropout_rate),
                                       lambda: c_rep)
            assert c_rep.get_shape()[-1].value == hps.size
        with tf.variable_scope('iterative_aligner'):
            for i in xrange(hps.T):
                with tf.variable_scope('question_aware_%d' % i):
                    with tf.variable_scope('multihead_attention'):
                        context_q = multihead_attention(c_rep, q_rep)
                    with tf.variable_scope('gate'):
                        inputs = gate(c_rep, context_q)
                    with tf.variable_scope('GRU'):
                        c_rep, c_state = bi_sru(
                            x=inputs,
                            output_size=hps.size,
                            sequence_length=self.context_lens,
                            dtype=self.dtype
                        )
                        c_rep = tf.concat(c_rep, axis=-1)
                        c_rep = tf.layers.dense(c_rep, units=hps.size, use_bias=False, name='c_rep')
                        c_rep = tf.cond(self.dropout, lambda: tf.nn.dropout(c_rep, keep_prob=1.0 - hps.dropout_rate),
                                        lambda: c_rep)
                with tf.variable_scope('self_attention_%d' % i):
                    with tf.variable_scope('multihead_attention'):
                        context_c = multihead_attention(c_rep, c_rep)
                    with tf.variable_scope('semantic_fusion_unit'):
                        inputs = gate(c_rep, context_c)
                    with tf.variable_scope('GRU'):
                        c_rep, c_state = bi_sru(
                            x=inputs,
                            output_size=hps.size,
                            sequence_length=self.context_lens,
                            dtype=self.dtype
                        )
                        c_rep = tf.concat(c_rep, axis=-1)
                        c_rep = tf.layers.dense(c_rep, units=hps.size, use_bias=False, name='c_rep')
                        c_rep = tf.cond(self.dropout, lambda: tf.nn.dropout(c_rep, keep_prob=1.0 - hps.dropout_rate),
                                        lambda: c_rep)
                        assert c_rep.get_shape()[-1].value == hps.size
        with tf.variable_scope('output_layer'):
            with tf.variable_scope('init_state'):
                z_s = tf.layers.dense(tf.concat(q_state, axis=-1), units=hps.size, use_bias=False, name='z_s')
                z_s = tf.expand_dims(z_s, axis=1)
                z_s = tf.cond(self.dropout, lambda: tf.nn.dropout(z_s, keep_prob=1.0 - hps.dropout_rate), lambda: z_s)
            for i in xrange(hps.T):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                with tf.variable_scope('start_position'):
                    start_pos_scores, u_s = fn(c_rep, z_s)
                with tf.variable_scope('start_pos_memory_semantic_fusion_unit'):
                    z_e = sfu(z_s, u_s)
                    z_e = tf.cond(self.dropout, lambda: tf.nn.dropout(z_e, keep_prob=1.0 - hps.dropout_rate),
                                  lambda: z_e)
                with tf.variable_scope('end_position'):
                    end_pos_scores, u_e = fn(c_rep, z_e)
                with tf.variable_scope('end_pos_memory_semantic_fusion_unit'):
                    z_s = sfu(z_e, u_e)
                    z_s = tf.cond(self.dropout, lambda: tf.nn.dropout(z_s, keep_prob=1.0 - hps.dropout_rate),
                                  lambda: z_s)
            self.pos_scores = [start_pos_scores, end_pos_scores]

    def set_death_rate(self, l, L, p):
        return l / L * (1 - p)

    def _loss(self):
        with tf.name_scope('log_perplexity'):
            log_perplexity_list = []
            for score, ans in zip(self.pos_scores, self.answer):
                probability = tf.nn.softmax(score)
                entropy = tf.log(tf.clip_by_value(probability, 1e-10, 1.0))
                y_ = tf.one_hot(indices=ans, depth=self.max_c_length)
                crossent = -tf.reduce_mean(tf.reduce_sum(y_ * entropy, axis=-1))
                log_perplexity_list.append(crossent)
            log_perplexity = tf.add_n(log_perplexity_list) / 2
        with tf.name_scope('l2_regularization'):
            self.tvars = tf.trainable_variables()
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self._hps.weight_decay_rate)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer=l2_regularizer, weights_list=self.tvars)
        self.cost = log_perplexity + l2_loss
        tf.summary.scalar('cost', self.cost)

    def train_op(self):
        with tf.device('/gpu:0'):
            grad, global_norm = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars), self._hps.max_grad_norm)
        tf.summary.scalar('global_norm', global_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        tf.summary.scalar('learning_rate', self.lr)
        self._train_op = optimizer.apply_gradients(zip(grad, self.tvars), global_step=self.global_step, name='train_step')

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

    def run_trian_step(self, sess, context_words, question_words, context_chars, question_chars, context_lens,
                       question_lens, answer, batch_size, max_c_length, max_q_length, max_char_length, dropout):
        input_feed = self.feed_data(context_words, question_words, context_chars, question_chars, context_lens,
                                    question_lens, answer, batch_size, max_c_length, max_q_length, max_char_length,
                                    dropout)
        output_feed = [self._train_op, self.cost, self.pos_scores, self._summary, self.global_step]
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs[1:]

    def run_eval_step(self, sess, context_words, question_words, context_chars, question_chars, context_lens,
                      question_lens, answer, batch_size, max_c_length, max_q_length, max_char_length, dropout):
        input_feed = self.feed_data(context_words, question_words, context_chars, question_chars, context_lens,
                                    question_lens, answer, batch_size, max_c_length, max_q_length, max_char_length,
                                    dropout)
        output_feed = [self.cost, self.pos_scores, self._summary, self.global_step]
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs

    def run_decode_step(self, sess, context_words, question_words, context_chars, question_chars, context_lens,
                        question_lens, answer, batch_size, max_c_length, max_q_length, max_char_length, dropout):
        input_feed = self.feed_data(context_words, question_words, context_chars, question_chars, context_lens,
                                    question_lens, answer, batch_size, max_c_length, max_q_length, max_char_length,
                                    dropout)
        output_feed = [self.pos_scores, self.global_step]
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs

    def feed_data(self, context_words, question_words, context_chars, question_chars, context_lens, question_lens,
                  answer, batch_size, max_c_length, max_q_length, max_char_length, dropout):
        input_feed = {}
        input_feed[self.batch_size.name] = batch_size
        input_feed[self.max_c_length.name] = max_c_length
        input_feed[self.max_q_length.name] = max_q_length
        input_feed[self.max_char_length.name] = max_char_length
        input_feed[self.dropout.name] = dropout
        input_feed[self.context_lens.name] = context_lens
        input_feed[self.question_lens.name] = question_lens
        input_feed[self.context_word_inputs.name] = context_words
        input_feed[self.question_word_inputs.name] = question_words
        input_feed[self.context_char_inputs.name] = context_chars
        input_feed[self.question_char_inputs.name] = question_chars
        for i in xrange(2):
            input_feed[self.answer[i].name] = answer[i]
        return input_feed
