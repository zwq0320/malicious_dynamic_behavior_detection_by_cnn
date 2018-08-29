import tensorflow as tf
import numpy as np
from gensim.models.word2vec import Word2Vec


DEFAULT_PADDING = 'VALID'


def load_word_vec(word_vec_npy, sess):
    word_vec = np.load(word_vec_npy)
    with tf.variable_scope("embedding", reuse=True):
        sess.run(tf.get_variable("weights").assign(word_vec))


def init_weight(x, sess, embedding_size, vocab_size):
    print("start init weight")
    x_str_list = [list(map(str, each_x)) for each_x in x.tolist()]
    print("len(x_str_list): %s" % len(x_str_list))
    model = Word2Vec(x_str_list, min_count=0, size=embedding_size)
    print("vocab_size: %s" % vocab_size)
    print("word_vec size: {}".format(model.wv.vectors.shape))
    word_vec_list = []
    for i in range(vocab_size):
        if str(i) not in model.wv:
            print("error: %s not in wv" % i)
            word_vec_list.append([0 for _ in range(100)])
        else:
            word_vec_list.append(model.wv[str(i)])
    word_vec = np.array(word_vec_list)
    print("word_vec[0]: %s" % word_vec[0])
    with tf.variable_scope("embedding", reuse=True):
        W = tf.get_variable("weights")
        print("init weight name: {}".format(W.name))
        sess.run(W.assign(word_vec))


def embedding(input_x, vocab_size, embedding_size, name):
    with tf.variable_scope(name):
        W = tf.get_variable("weights", [vocab_size, embedding_size])
        print("embedding W name: {}".format(W.name))
        embedded_chars = tf.nn.embedding_lookup(W, input_x)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        return embedded_chars_expanded


def conv(input, k_h, k_w, c_i, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING):
    with tf.variable_scope(name) as scope:
        filter_shape = [k_h, k_w, c_i, c_o]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="weights")
        b = tf.Variable(tf.constant(0.1, shape=[c_o]), name="biases")
        conv2d = tf.nn.conv2d(
            input,
            W,
            strides=[1, s_h, s_w, 1],
            padding=padding,
            name="conv")
        if relu:
            bias = tf.nn.bias_add(conv2d, b)
            return tf.nn.relu(bias, name=scope.name)
        return tf.reshape(tf.nn.bias_add(conv2d, b), conv2d.get_shape().as_list(), name=scope.name)


def relu(input, name):
    return tf.nn.relu(input, name=name)


def max_pool(input, k_h, k_w, s_w, s_h, name, padding=DEFAULT_PADDING):
    pooled = tf.nn.max_pool(
        input,
        ksize=[1, k_h, k_w, 1],
        strides=[1, s_w, s_h, 1],
        padding=padding,
        name=name)
    return pooled


def concat(inputs, axis, name):
    return tf.concat(inputs, axis, name=name)


def fc(input, num_in, num_out, l2_lambda, name):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable(
            "weights",
            shape=[num_in, num_out],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_out]), name="biases")
        f = tf.nn.xw_plus_b(input, W, b, name=scope.name)
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(l2_lambda)(W))
        return f


def batch_norm(input, training, name):
    return tf.layers.batch_normalization(input, training=training, name=name)


def dropout(input, dropout_keep_prob):
    return tf.nn.dropout(input, dropout_keep_prob)

