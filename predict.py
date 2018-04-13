#coding=utf-8
import tensorflow as tf
import time
import math
from mrc import Model
from params import Params as hps
from process import *
from evaluate import evaluate
from evaluate import exact_match_score
from evaluate import f1_score

tf.app.flags.DEFINE_string("mode", 'train', "Training model.")
FLAGS = tf.app.flags.FLAGS

def create_model(sess, hps, logdir, word_emb, char_emb):
    model = Model(hps, word_emb, char_emb)
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reading model parameters from %s' % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    return model

def _Train(hps, word_emb, char_emb):
    if hps.model == 'r_net':
        print('R_Net Model!!!!!!!!!')
        logdir = hps.logdir_r_net
    elif hps.model == 'mnemonic_reader':
        print('Mnemonic_Reader Model!!!!!!!!!')
        logdir = hps.logdir
    else:
        print('Transformer Model!!!!!!!!!')
        logdir = hps.logdir_transformer
    os.makedirs(hps.train_dir + hps.model, exist_ok=True)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, hps, logdir, word_emb, char_emb)
        train_summary_writer = tf.summary.FileWriter(hps.train_dir + hps.model)
        train_summary_writer.add_graph(sess.graph)
        c_word_dir = hps.train_dir + hps.c_word_dir
        q_word_dir = hps.train_dir + hps.q_word_dir
        c_chars_dir = hps.train_dir + hps.c_chars_dir
        q_chars_dir = hps.train_dir + hps.q_chars_dir
        target_dir = hps.train_dir + hps.target_dir
        train_data = data_queue(c_word_dir, q_word_dir, c_chars_dir, q_chars_dir, target_dir)

        c_word_dir = hps.dev_dir + hps.c_word_dir
        q_word_dir = hps.dev_dir + hps.q_word_dir
        c_chars_dir = hps.dev_dir + hps.c_chars_dir
        q_chars_dir = hps.dev_dir + hps.q_chars_dir
        target_dir = hps.dev_dir + hps.target_dir
        dev_data = data_queue(c_word_dir, q_word_dir, c_chars_dir, q_chars_dir, target_dir)
        dev_summary_writer = tf.summary.FileWriter(hps.dev_dir)

        step_time, loss = 0.0, 0.0
        current_step = 0
        while current_step < hps.max_run_steps:
            start_time = time.time()
            train_data, batch_c_word, batch_q_word, batch_c_chars, batch_q_chars, batch_target, batch_c_lens, \
                batch_q_lens, max_c_length, max_q_length, max_char_length = get_batch(train_data)
            print('-', end='')
            step_loss, pos_scores, summary, global_step = model.run_trian_step(sess, batch_c_word, batch_q_word,
                                                                               batch_c_chars,
                                                                               batch_q_chars, batch_c_lens,
                                                                               batch_q_lens,
                                                                               batch_target,
                                                                               hps.batch_size,
                                                                               max_c_length,
                                                                               max_q_length,
                                                                               max_char_length,
                                                                               dropout=True)
            train_summary_writer.add_summary(summary, global_step)
            step_time += (time.time() - start_time) / hps.steps_per_checkpoint
            loss += step_loss / hps.steps_per_checkpoint
            current_step += 1
            if current_step % hps.steps_per_checkpoint == 0:
                print('')
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(logdir, "predict.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=global_step)
                # print statistics for the previous epoch
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print("global step %d learning rate %.4f step-time %.2f perplexity %.2f"
                      % (global_step, model.lr.eval(), step_time, perplexity))

                em, f1 = EM_F1(pos_scores, batch_target)
                print('EM: %.2f, F1: %.2f' % (em, f1))

                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                dev_data, batch_c_word, batch_q_word, batch_c_chars, batch_q_chars, batch_target, batch_c_lens, \
                    batch_q_lens, max_c_length, max_q_length, max_char_length = get_batch(dev_data)
                eval_loss, pos_scores, summary, global_step = model.run_eval_step(sess, batch_c_word, batch_q_word,
                                                                                  batch_c_chars, batch_q_chars,
                                                                                  batch_c_lens, batch_q_lens,
                                                                                  batch_target,
                                                                                  hps.batch_size,
                                                                                  max_c_length,
                                                                                  max_q_length,
                                                                                  max_char_length,
                                                                                  dropout=False)
                #dev_summary_writer.add_summary(summary, global_step)
                eval_ppx = math.exp(float(eval_loss)) if loss < 300 else float("inf")
                print("  eval: perplexity %.2f" % eval_ppx)

                em, f1 = EM_F1(pos_scores, batch_target)
                print('  EM: %.2f, F1: %.2f' % (em, f1))
            sys.stdout.flush()

def EM_F1(pos_scores, batch_target):
    pos = [np.argmax(x, axis=1) for x in pos_scores]
    predict_ans = normalize_ans(pos)
    ans = normalize_ans(batch_target)
    em = f1 = 0
    for prediction, ground_truth in zip(predict_ans, ans):
        em += exact_match_score(prediction, ground_truth)
        f1 += f1_score(prediction, ground_truth)
    em = 100.0 * em / len(ans)
    f1 = 100.0 * f1 / len(ans)
    return em, f1

def normalize_ans(ans):
    _ans = []
    for s, e in zip(*ans):
        if s == e:
            _ans.append(' '.join([str(s), str(e)]))
        else:
            _ans.append(' '.join([str(x) for x in list(range(int(s), int(e) + 1))]))
    return _ans

def main(_):
    process_data()
    word_emb = np.memmap(hps.data_dir + "glove.np", dtype=np.float32, mode="r")
    word_emb = np.reshape(word_emb, (-1, hps.emb_size))
    print('word_emb ok!!!')
    char_emb = np.memmap(hps.data_dir + "glove_char.np", dtype=np.float32, mode="r")
    char_emb = np.reshape(char_emb, (hps.char_vocab_size, hps.char_emb_size))
    print('char_emb ok!!!')
    hps.mode = FLAGS.mode
    if FLAGS.mode == 'train':
        _Train(hps, word_emb, char_emb)

if __name__ == '__main__':
    tf.app.run()