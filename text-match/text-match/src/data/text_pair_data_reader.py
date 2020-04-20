#coding: utf-8
import copy
import os, sys
import random

import numpy as np
import pandas as pd
import tensorflow as tf

QUERY_WORD_COUNT = 20
TITLE_WORD_COUNT = 20


def local_data(path, batch_size, repeat_count, word_count, neg_count = 4, cycle_length = 8, sloppy = True):
    filenames = []
    for f in os.listdir(path):
        filenames.append('%s/%s' % (path, f))
    print("filenames: ", filenames)
    files_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    ds = files_dataset.apply(
                tf.data.experimental.parallel_interleave(
                    lambda file_name: _text_generator(file_name, word_count),
                    cycle_length = cycle_length, 
                    sloppy = sloppy, 
                    buffer_output_elements = 1024    
                )
            )
    ds = ds.shuffle(100 * batch_size).batch(batch_size).repeat()
    #ds = ds.padded_batch(batch_size, ([None], [None]))
    #ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    #ds = ds.repeat()
    #it = iter(ds)
    #while True:
    #    try:
    #        label, query, pos_data, neg_data = next(it)
    #        features = tuple([query, pos_data, neg_data])
    #        yield features, label
    #    except Exception as e:
    #        it = iter(ds)
    #        print('epoch [%d] run out' % (i))
    return ds


def _text_generator(file_name, word_count, neg_count = 4):
    return tf.data.Dataset.from_generator(
                _text_handler,
                output_types = ({"query": tf.int64, "pos_data": tf.int64, "neg_data": tf.int64}, tf.int64),
                output_shapes = ({"query": tf.TensorShape([None]), "pos_data": tf.TensorShape([None]), "neg_data": tf.TensorShape([None])}, tf.TensorShape([None])),
                args = (file_name, word_count, neg_count)
            )

def _text_handler(file_name, word_count, neg_count = 4):
    """
    data format is:
    label\tquery\tpos_data\tneg_data
    label(int): 1 or 0
    query(id list): 1,2,3,4,5 
    pos_data(id list): 1,2,3,4,5 
    neg_data(id list:id_list): 1,2,3:5,6,7:8,9,10
    """
    values = pd.read_csv(file_name.decode('utf-8'), sep = '\t').values
    for value in values:
        #if len(value) != 4:
        #    continue
        #label = np.zeros(neg_count + 1)
        #label[0] = 1
        label = tf.keras.utils.to_categorical(0, 2)
        query = _format_text_value(value[1], word_count)
        pos_data = _format_text_value(value[2], word_count)
        neg_data = _format_text_value(value[3], word_count)

        #n = random.randint(0, neg_count-1)
        #neg_data = _format_text_value(value[3].split(':')[n], word_count)
        #neg_data = np.array([ _format_text_value(v, word_count) for v in value[3].split(':') ])
        #yield (query, pos_data, neg_data), label
        yield {"query": query, "pos_data": pos_data, "neg_data": neg_data}, label

def _format_text_value(value, word_count):
    t = np.array([ int(v) for v in value.split(',') ])
    pad_len = word_count - len(t)
    if pad_len < 0:
        return t[:word_count]
    else:
        return np.pad(t, (0, pad_len), 'constant')


if __name__ == '__main__':
    with tf.compat.v1.Session() as session:
        #a = tf.constant(1.0)
        #b = tf.constant(6.0)
        #c = a * b
        ds = local_data(sys.argv[1], 1, 1, 10)
        #dd = ds * a
        a,b,c,d = session.run(next(ds))
        print('a: ', a)
        #print(features)
