# -*- coding: utf-8 -*-
#/usr/bin/python2

from functools import wraps
import threading
from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf
from process import *

# Adapted from the `sugartensor` code.
# https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def producer_func(func):
    r"""Decorates a function `func` as producer_func.
    Args:
      func: A function to decorate.
    """
    @wraps(func)
    def wrapper(inputs, dtypes, capacity, num_threads):
        r"""
        Args:
            inputs: A inputs queue list to enqueue
            dtypes: Data types of each tensor
            capacity: Queue capacity. Default is 32.
            num_threads: Number of threads. Default is 1.
        """
        # enqueue function
        def enqueue_func(sess, op):
            # read data from source queue
            data = func(sess.run(inputs))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)

        # create place holder list
        placeholders = []
        for dtype in dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))

        # create FIFO queue
        queue = tf.FIFOQueue(capacity, dtypes=dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op] * num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)

        # return de-queue operation
        return queue.dequeue()

    return wrapper


class _FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                               queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op)  # call enqueue function
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1

def load_data(dir_):
    # Target indices
    indices = load_target(dir_ + hps.target_dir)

    # Load question data
    print("Loading question data...")
    q_word_ids, _ = load_word(dir_ + hps.q_word_dir)
    q_char_ids, q_char_len, q_word_len = load_char(dir_ + hps.q_chars_dir)

    # Load context data
    print("Loading context data...")
    c_word_ids, _ = load_word(dir_ + hps.p_word_dir)
    c_char_ids, c_char_len, c_word_len = load_char(dir_ + hps.c_chars_dir)

    # Get max length to pad
    c_max_word = hps.max_c_len  #np.max(c_word_len)
    c_max_char = hps.max_char_len  #max_value(c_char_len))
    q_max_word = hps.max_q_len  #np.max(q_word_len)
    q_max_char = hps.max_char_len  #max_value(q_char_len))

    # pad_data
    print("Preparing data...")
    c_word_ids = pad_data(c_word_ids, c_max_word)
    q_word_ids = pad_data(q_word_ids, q_max_word)
    c_char_ids = pad_char_data(c_char_ids, c_max_char, c_max_word)
    q_char_ids = pad_char_data(q_char_ids, q_max_char, q_max_word)

    # to numpy
    indices = np.reshape(np.asarray(indices, np.int32), (-1, 2))
    c_word_len = np.reshape(np.asarray(c_word_len, np.int32), (-1, 1))
    q_word_len = np.reshape(np.asarray(q_word_len, np.int32), (-1, 1))

    # shapes of each data
    shapes = [(c_max_word,), (q_max_word,),
             (c_max_word, c_max_char,), (q_max_word, q_max_char,),
             (1,), (1,),
             (2,)]

    return ([c_word_ids, q_word_ids,
            c_char_ids, q_char_ids,
            c_word_len, q_word_len,
            indices], shapes)

def get_dev():
    devset, shapes = load_data(hps.dev_dir)
    indices = devset[-1]
    # devset = [np.reshape(input_, shapes[i]) for i,input_ in enumerate(devset)]

    dev_ind = np.arange(indices.shape[0], dtype=np.int32)
    np.random.shuffle(dev_ind)
    return devset, dev_ind

def get_batch(is_training = True):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load dataset
        input_list, shapes = load_data(hps.train_dir if is_training else hps.dev_dir)
        indices = input_list[-1]
        size = indices.shape[0]
        train_ind = np.arange(size, dtype=np.int32)
        np.random.shuffle(train_ind)

        ind_list = tf.convert_to_tensor(train_ind)

        # Create Queues
        ind_list = tf.train.slice_input_producer([ind_list], shuffle=True)

        @producer_func
        def get_data(ind):
            '''From `_inputs`, which has been fetched from slice queues,
               then enqueue them again.
            '''
            return [np.reshape(input_[ind], shapes[i]) for i, input_ in enumerate(input_list)]

        data = get_data(inputs=ind_list,
                        dtypes=[np.int32]*9,
                        capacity=hps.batch_size*32,
                        num_threads=6)

        # create batch queues
        batch = tf.train.batch(data,
                                shapes=shapes,
                                num_threads=2,
                                batch_size=hps.batch_size,
                                capacity=hps.batch_size*32,
                                dynamic_pad=True)

    return batch, size // hps.batch_size
