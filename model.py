import tensorflow as tf
from network import embedding, conv, max_pool, concat, dropout, fc, batch_norm


class Model(object):
    @staticmethod
    def cnn_text(net_input, sequence_length, vocab_size, embedding_size, stride_h, filter_sizes, num_filters,
                 num_classes, keep_prob, l2_lambda, is_bn, bn_training):
        embedding_layer = embedding(net_input, vocab_size, embedding_size, "embedding")
        pools = list()
        for i, filter_size in enumerate(filter_sizes):
            convi = conv(embedding_layer, filter_size, embedding_size, 1, num_filters, stride_h, 1, "conv" + str(i))
            if is_bn:
                bni = batch_norm(convi, bn_training, "bn" + str(i))
                convi = bni
            pooli = max_pool(convi, (sequence_length - filter_size) // stride_h + 1, 1, 1, 1, "pool" + str(i))
            pools.append(pooli)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = concat(pools, 3, "concat")
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        h_drop = dropout(h_pool_flat, keep_prob)
        h_fc = fc(h_drop, num_filters_total, num_classes, l2_lambda, "fc")
        return h_fc
