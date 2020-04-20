#coding: utf-8
import copy
import os, sys

import numpy as np
import pandas as pd
import tensorflow as tf

QUERY_WORD_COUNT = 20
TITLE_WORD_COUNT = 20

def local_data(path, batch_size, epochs, word_count, cycle_length = 8, sloppy = True):
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
    ds = ds.repeat(epochs)
    ds = ds.shuffle(10 * batch_size).batch(batch_size)
    ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    #it = ds.make_one_shot_iterator()
    #label, query, data = it.get_next()
    #label, query, data = next(ds.__iter__())
    #it = iter(ds)
    #while True:
    #    query, data, label = next(it)
    #    features = (query, data)
    #    yield features, label
    #return (features, label)
    return ds


def _text_generator(file_name, word_count):
    return tf.data.Dataset.from_generator(
                _text_handler,
                output_types = ({"query": tf.int64, "pos_data": tf.int64}, tf.int64),
                output_shapes = ({"query": tf.TensorShape([None]), "pos_data": tf.TensorShape([None])}, tf.TensorShape([None])),
                args = (file_name, word_count, neg_count)
            )

def _text_handler(file_name, query_word_count = 20, doc_word_count = 20):
    """
    data format is:
    label\tquery\tpos_data\tneg_data
    label(int): 1 or 0
    query(id list): 1,2,3,4,5 
    data(id list): 1,2,3,4,5 
    """
    values = pd.read_csv(file_name.decode('utf-8'), sep = '\t').values
    for value in values:
        #if len(value) != 3:
        #    continue
        label = int(value[0])
        query = _format_text_value(str(value[1]), query_word_count)
        data = _format_text_value(str(value[2]), doc_word_count)
        yield {"query": query, "pos_data": pos_data}, label

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
        a,b,c = session.run(next(ds))
        print('a: ', a)
        #print(features)
