import tensorflow as tf
from model import Model
from network import init_weight
from data import load_data_and_labels, data_iter
import datetime
import numpy as np
from tensorflow.contrib import learn
import os


def preprocess(data_dirs, document_length_limit, is_line_as_word, dev_sample_percentage):

    x_text, y = load_data_and_labels(data_dirs, document_length_limit, is_line_as_word)

    # Vocabulary
    max_document_length = max([len(text.split(" ")) for text in x_text])
    print("max_docment_length: {}".format(max_document_length))
    max_document_length = min(document_length_limit, max_document_length)
    print("max_docment_length: {}".format(max_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print(x)

    # Random
    np.random.seed(100)
    shuffle_indices = np.random.permutation(np.arange(len(x_text)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(dev_sample_percentage * len(y))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled
    
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


def train(x_train, y_train, vocab_processor, x_dev, y_dev,
          num_classes, embedding_size, filter_sizes, stride_h, num_filters, keep_prob_rate, learning_rate_val,
          decay_steps, decay_rate, l2_lambda, batch_size, ecoph_num, evaluate_every, checkpoint_every,
          is_finetune, is_bn,
          out_dir):
    # Paras
    sequence_length = x_train.shape[1]
    vocab_size = len(vocab_processor.vocabulary_)

    # Graph input
    x_input = tf.placeholder(tf.int32, [None, sequence_length], name="x_input")
    y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
    keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    is_training = tf.placeholder(tf.bool, name="mode")

    # Step
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Model
    pred = Model.cnn_text(x_input, sequence_length, vocab_size, embedding_size, stride_h, filter_sizes, num_filters,
                          num_classes, keep_prob, l2_lambda, is_bn, is_training)
    # Loss
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_input)
    loss = tf.reduce_mean(losses)
    if l2_lambda > 0:
        tf.add_to_collection("losses", loss)
        loss = tf.add_n(tf.get_collection("losses"))   # if l2=0, then the first value of tf.get_collection will be None

    # Learning rate decay
    learning_rate = tf.train.exponential_decay(learning_rate=learning_rate_val, global_step=global_step,
                                               decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Evaluation
    predictions = tf.argmax(pred, 1, name="predictions")   # needed in test
    correct_pred = tf.equal(predictions, tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")

    # Init
    init = tf.global_variables_initializer()

    # Summary
    dev_ratio = x_dev.shape[0] * 1.0 / (x_train.shape[0] + x_dev.shape[0])
    train_summary_path = os.path.join(out_dir, "summaries", "train")
    dev_summary_path = os.path.join(out_dir, "summaries", "dev")
    if not os.path.exists(train_summary_path):
        os.makedirs(train_summary_path)
    if not os.path.exists(dev_summary_path):
        os.makedirs(dev_summary_path)
    train_summary_writer = tf.summary.FileWriter(train_summary_path)
    dev_summary_writer = tf.summary.FileWriter(dev_summary_path)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    loss_summary = tf.summary.scalar("loss", loss)
    acc_summary = tf.summary.scalar("accuracy", accuracy)
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])

    # Saver
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=5)

    # Write vocabulary
    vocab_processor.save(os.path.join(out_dir, "vocab"))

    def train_step(x_batch, y_batch):
        feed_dict = {
            x_input: x_batch,
            y_input: y_batch,
            keep_prob: keep_prob_rate,
            is_training: True
        }
        _, step, summaries, train_loss, train_accuracy = sess.run(
            [train_op, global_step, train_summary_op, loss, accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("[train] {}: step {}, loss {}, acc {}".format(time_str, step, train_loss, train_accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch):
        feed_dict = {
            x_input: x_batch,
            y_input: y_batch,
            keep_prob: 1.0,    # no dropout when test
            is_training: False
        }
        step, summaries, test_loss, test_accuracy = sess.run(
            [global_step, dev_summary_op, loss, accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("[test] {}: step {}, loss {}, acc {}".format(time_str, step, test_loss, test_accuracy))
        dev_summary_writer.add_summary(summaries, step)

    with tf.Session() as sess:
        sess.run(init)

        train_summary_writer.add_graph(sess.graph)
        dev_summary_writer.add_graph(sess.graph)

        if is_finetune:
            print("Load pre-trained word vector")
            init_weight(np.concatenate([x_train, x_dev]), sess, embedding_size, vocab_size)

        batches = data_iter(zip(x_train, y_train), batch_size, ecoph_num)
        for batch_x_y in batches:
            x_batch, y_batch = zip(*batch_x_y)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev)
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


def main():
    # Paras
    positive_data_dir = "data/train/pos"
    negative_data_dir = "data/train/neg"
    data_dirs = [negative_data_dir, positive_data_dir]
    out_dir = "runs"
    document_length_limit = 1000
    is_line_as_word = True
    dev_sample_percentage = 0.1
    num_classes = len(data_dirs)
    embedding_size = 100
    filter_sizes = [3, 4, 5]
    stride_h = 1
    num_filters = 128
    keep_prob_rate = 1.0
    learning_rate = 1e-3
    batch_size = 32
    ecoph_num = 100
    evaluate_every = 10
    checkpoint_every = 10
    is_finetune = False
    is_bn = True   # use batch norm or not
    l2_lambda = 0.0
    decay_steps = 1000
    decay_rate = 0.5

    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess(data_dirs,
                                                                 document_length_limit, 
                                                                 is_line_as_word, 
                                                                 dev_sample_percentage)
    train(x_train, y_train, vocab_processor, x_dev, y_dev,
          num_classes, embedding_size, filter_sizes, stride_h, num_filters, keep_prob_rate, learning_rate, decay_steps,
          decay_rate, l2_lambda, batch_size, ecoph_num, evaluate_every, checkpoint_every,
          is_finetune, is_bn, out_dir)


if __name__ == "__main__":
    main()











