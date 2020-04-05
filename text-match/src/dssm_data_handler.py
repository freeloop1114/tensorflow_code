#coding: utf-8
from __future__ import print_function
import os
import pandas as pd
import sys

import numpy as np
import tensorflow as tf
reload(sys)
sys.setdefaultencoding('utf-8')

######## API: convert data to record ##########
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _int64_matrix_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))

def _int64_matrix_feature_shape(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value.shape))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _format_label(value):
    return int(value)

def _format_pos(value):
    return np.array([int(i.strip()) for i in value.split(',')], dtype=np.int64)

def _format_neg(value):
    tmp = []
    for v in value.split(":"):
        tmp.append([int(i.strip()) for i in v.split(',')])
    return np.array(tmp, dtype=np.int64)
        
def convert(input_file, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    input = pd.read_csv(input_file).values
    for value in input:
            label = _format_label(value[0])
            query = _format_pos(value[1])
            pos = _format_pos(value[2])
            neg = _format_neg(value[3])
            features = {
                "label": _int64_feature(label),
                "query": _int64_list_feature(query),
                "pos": _int64_list_feature(pos),
                "neg": _int64_matrix_feature(neg),
                "neg_shape": _int64_matrix_feature_shape(neg)
            }
            example = tf.train.Example(
                    features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
    writer.close()

######## API: parse record ##########
def parse_record(example_proto):
    features = {
        "label": tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
        "query": tf.VarLenFeature(dtype=tf.int64),
        "pos": tf.VarLenFeature(dtype=tf.int64),
        "neg": tf.VarLenFeature(dtype=tf.int64),
        "neg_shape": tf.FixedLenFeature(shape=(2,), dtype=tf.int64, default_value=None)
    }

    parsed_example = tf.parse_single_example(example_proto, features=features)
    return parsed_example


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
    dataset = tf.data.TFRecordDataset(sys.argv[2])
    dataset = dataset.map(parse_record)
    iterator = dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        features = sess.run(iterator.get_next())
        features['neg'] = tf.sparse_tensor_to_dense(features['neg'])
        features['neg'] = tf.reshape(features['neg'], features['neg_shape'])
        print(features['neg'])
