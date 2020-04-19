#coding: utf-8
import sys
sys.path.append("..")

import numpy as np
import scipy as sp
import tensorflow as tf
from models.base_model import BaseModel

class TextDssmModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)

    def build_graph(self):
        # 1. input data
        query_input = tf.keras.Input(shape = (self.word_count, ), dtype = tf.int64, name = "query")
        pos_input = tf.keras.Input(shape = (self.word_count, ), dtype = tf.int64, name = "pos_data")
        neg_input = tf.keras.Input(shape = (self.word_count, ), dtype = tf.int64, name = "neg_data")
        masking = tf.keras.layers.Masking(mask_value = 0)
        query_input = masking(query_input)
        pos_input = masking(pos_input)
        neg_input = masking(neg_input)

        # 2. network
        query, pos_doc, neg_doc = self.word_embedding(query_input, pos_input, neg_input)
        query, pos_doc, neg_doc = self.mlp(query, pos_doc, neg_doc)
        concat_sim = self.similarity(query, pos_doc, neg_doc)
        prob = tf.keras.layers.Softmax()(concat_sim)
        print("prob: ", prob.shape)

        # 3. model and loss
        self.model = tf.keras.Model(inputs = [query_input, pos_input, neg_input], outputs = prob)
        self.model.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.CategoricalAccuracy()]
        )
        print(self.model.summary())

    
    def word_embedding(self, query_input, pos_input, neg_input):
        word = tf.keras.layers.Embedding(
            input_dim = self.word_size,
            output_dim = self.word_embedding_size,
            embeddings_initializer = 'glorot_normal',
            mask_zero = True
        )

        query = tf.keras.backend.sum(word(query_input), axis = 1)
        pos_doc = tf.keras.backend.sum(word(pos_input), axis = 1)
        neg_doc = tf.keras.backend.sum(word(neg_input), axis = 1)
        print("query: ", query.shape)
        return query, pos_doc, neg_doc


    def mlp(self, query, pos_doc, neg_doc):
        for i, layer in enumerate(self.mlp_layers):
            query_mlp = tf.keras.layers.Dense(
                            layer,
                            activation = 'relu', 
                            kernel_initializer = 'glorot_normal'
                        )
            doc_mlp = tf.keras.layers.Dense(
                            layer,
                            activation = 'relu', 
                            kernel_initializer = 'glorot_normal'
                        )
            query_with_bn = tf.keras.layers.BatchNormalization()
            doc_with_bn = tf.keras.layers.BatchNormalization()
            query = query_mlp(query)
            pos_doc = doc_mlp(pos_doc)
            neg_doc = doc_mlp(neg_doc)
            #query = query_with_bn(query)
            #pos_doc = doc_with_bn(pos_doc)
            #neg_doc = doc_with_bn(neg_doc)
            #neg_doc = [doc_mlp(neg_doc) for neg_doc in neg_doc]
            print("[MLP_%d] query: " % (i), query.shape)
            print("[MLP_%d] pos_doc: " % (i), pos_doc.shape)
        return query, pos_doc, neg_doc


    def similarity(self, query, pos_doc, neg_doc):
        dot = tf.keras.layers.Dot(axes = 1, normalize = True)
        query_pos_sim = dot([query, pos_doc])
        query_neg_sim = dot([query, neg_doc])
        concat_sim = tf.keras.layers.concatenate([query_pos_sim, query_neg_sim])
        print("[SIM] concat_sim: ", concat_sim.shape)
        return concat_sim
