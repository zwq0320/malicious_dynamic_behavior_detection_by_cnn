import os
import re
import csv
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
from data import load_data_label_and_filenames, data_iter


def test(data_dirs, checkpoint_dir, document_length_limit, batch_size, is_line_as_word):
    # Restore vocab
    vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    # Load data
    x_raw, y_test, filenames = load_data_label_and_filenames(data_dirs, document_length_limit, is_line_as_word)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    y_test = np.argmax(y_test, axis=1)

    # Evaluation
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("x_input").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            mode = graph.get_operation_by_name("mode").outputs[0]

            fc = graph.get_operation_by_name("fc/fc").outputs[0]
            predictions = graph.get_operation_by_name("predictions").outputs[0]

            batches = data_iter(list(x_test), batch_size, 1, shuffle=False)

            all_predictions = []
            all_fc_scores = None

            for x_test_batch in batches:
                batch_fc_score, batch_predictions = sess.run([fc, predictions], {input_x: x_test_batch,
                                                             dropout_keep_prob: 1.0,
                                                             mode: False})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                if all_fc_scores is None:
                    all_fc_scores = batch_fc_score
                else:
                    all_fc_scores = np.concatenate([all_fc_scores, batch_fc_score])

    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

    # Save result
    predictions_human_readable = np.column_stack((np.array(filenames), all_predictions, y_test, all_fc_scores))
    out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, "w") as f:
        csv.writer(f).writerows(predictions_human_readable)

    wrong_predictions_human_readable = predictions_human_readable[all_predictions != y_test]
    wrong_path = os.path.join(checkpoint_dir, "..", "wrong.csv")
    print("Saving wrong evaluation to {0}".format(wrong_path))
    with open(wrong_path, "w") as f:
        csv.writer(f).writerows(wrong_predictions_human_readable)


def get_document_length_limit(dir_name):
    p = re.compile('sen_len=(\d+)')
    m = p.search(dir_name)
    if m:
        return int(m.group(1))
    else:
        return None


def main():
    pos_dir = "data/test/pos"
    neg_dir = "data/test/neg"
    data_dirs = [neg_dir, pos_dir]
    checkpoint_dir = "runs/checkpoints"
    batch_size = 64
    is_line_as_word = True
    document_length_limit = 1000

    test(data_dirs, checkpoint_dir, document_length_limit, batch_size, is_line_as_word)


if __name__ == "__main__":
    main()

