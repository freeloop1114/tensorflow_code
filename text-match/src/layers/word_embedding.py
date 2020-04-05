# coding: utf-8

import numpy as np
from keras.layers import *
from keras.models import *
import tensorflow as tf

def word_embedding(shape, dtype=tf.float32, name='word_embedding', initializer=tf.random_normal_initializer(stddev=0.1), trainable=True):
  with tf.compat.v1.variable_scope(name):
    return tf.compat.v1.get_variable('embedding', shape, dtype=dtype, initializer=initializer, trainable=trainable)

def sparse_text_embedding(inputs, shape, weights=None, name='sparse_text_embedding', initializer=tf.random_normal_initializer(stddev=0.1), trainable=True):
  with tf.compat.v1.variable_scope(name):
    embedding = word_embedding(shape, initializer=initializer, trainable=trainable)
    return tf.nn.relu(tf.nn.embedding_lookup_sparse(embedding, inputs, weights, combiner='sum'))

def dense_text_embedding(inputs, shape, weights=None, name='dense_text_embedding', initializer=tf.random_normal_initializer(stddev=0.1), trainable=True):
  with tf.compat.v1.variable_scope(name):
    embedding = word_embedding(shape, initializer=initializer, trainable=trainable)
    return tf.contrib.layers.embedding_lookup_unique(embedding, inputs)