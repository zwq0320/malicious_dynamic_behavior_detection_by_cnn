import os
import re
import hashlib
import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_examples_from_dir(data_dir, max_length, is_line_as_word=False):
    examples = []
    if not os.path.isdir(data_dir):
        return examples
    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)
        f = open(full_path, "r")
        data = f.read()
        line_num = len(data.split("\n"))
        if line_num < 5:
            continue
        if not is_line_as_word:
            examples.append(data.strip())
        else:
            lines = data.split("\n")
            # replace each line as md5
            words = [hashlib.md5(line).hexdigest() for line in lines]
            examples.append(" ".join(words[:max_length]))
        f.close()

    return examples


def get_example_filenames_from_dir(data_dir, max_length, is_line_as_word=False):
    examples = []
    filenames = []
    if not os.path.isdir(data_dir):
        return examples, filenames
    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)
        f = open(full_path, "r")
        data = f.read()
        line_num = len(data.split("\n"))
        new_lines = []
        for line in data.split("\n"):
            if not line.startswith("#"):
                new_lines.append(line)
        data = "\n".join(new_lines)
        if line_num < 5:
            continue
        filenames.append(full_path)
        if not is_line_as_word:
            examples.append(data.strip())
        else:
            lines = data.split("\n")
            # replace each line as md5
            words = [hashlib.md5(line).hexdigest() for line in lines]
            examples.append(" ".join(words[:max_length]))
        f.close()

    return examples, filenames


def load_data_and_labels(data_dirs, max_document_length, is_line_as_word):
    x_text = []
    y = []
    labels = np.eye(len(data_dirs), dtype=np.int32).tolist()
    for i, data_dir in enumerate(data_dirs):
        examples = get_examples_from_dir(data_dir, max_document_length, is_line_as_word)
        x_text += [clean_str(sent) for sent in examples]
        y += [labels[i]] * len(examples) 
    y = np.array(y)
    return [x_text, y]


def load_data_label_and_filenames(data_dirs, max_document_length, is_line_as_word):
    x_text = []
    y = []
    fnames = []
    labels = np.eye(len(data_dirs), dtype=np.int32).tolist()
    for i, data_dir in enumerate(data_dirs):
        examples, fname = get_example_filenames_from_dir(data_dir, max_document_length, is_line_as_word)
        x_text += [clean_str(sent) for sent in examples]
        y += [labels[i]] * len(examples)
        fnames += fname
    y = np.array(y)
    return x_text, y, fnames


def data_iter(data, batch_size, ecoph_num, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    batch_count = int((data_size - 1) / batch_size) + 1
    print("data_size: {}".format(data_size))
    print("batch_count: {}".format(batch_count))
    for e in range(ecoph_num):
        shuffle_data = data
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]

        for i in range(batch_count):
            yield shuffle_data[i * batch_size: (i + 1) * batch_size]
