#coding: utf-8
import sys
sys.path.append("..")

import numpy as np
import scipy as sp
import tensorflow as tf
from models.base_model import BaseModel

class TextDssmMlpSimilarityModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)

    def build_graph(self):
        # 1. input data
        query_input = tf.keras.Input(shape = (self.word_count, ), dtype = tf.int64, name = "query")
        pos_input = tf.keras.Input(shape = (self.word_count, ), dtype = tf.int64, name = "pos_data")
        neg_input = tf.keras.Input(shape = (self.word_count, ), dtype = tf.int64, name = "neg_data")
        masking = tf.keras.layers.Masking(mask_value = 0)
        query_mask = masking(query_input)
        pos_mask = masking(pos_input)
        neg_mask = masking(neg_input)

        # 2. network 
        query, pos_doc, neg_doc = self.word_embedding(query_mask, pos_mask, neg_mask)
        query_pos, query_neg = self.mlp(query, pos_doc, neg_doc)
        concat_sim = self.similarity(query_pos, query_neg)
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
        )

        query = tf.keras.backend.sum(word(query_input), axis = 1)
        pos_doc = tf.keras.backend.sum(word(pos_input), axis = 1)
        neg_doc = tf.keras.backend.sum(word(neg_input), axis = 1)
        print("query: ", query.shape)
        return query, pos_doc, neg_doc


    def mlp(self, query, pos_doc, neg_doc):
        query_pos = tf.concat([query, pos_doc], axis=-1)
        query_neg = tf.concat([query, neg_doc], axis=-1)
        for i, output_dim in enumerate(self.mlp_layers):
            layer = tf.keras.layers.Dense(
                            output_dim,
                            activation = 'relu', 
                            kernel_initializer = 'glorot_normal'
                        )
            query_pos = layer(query_pos)
            query_neg = layer(query_neg)
            print("[MLP_%d] query_pos: " % (i), query_pos.shape)
            print("[MLP_%d] query_neg: " % (i), query_neg.shape)
        return query_pos, query_neg


    def similarity(self, query_pos, query_neg):
        sim_mlp = tf.keras.layers.Dense(1)
        query_pos = sim_mlp(query_pos)
        query_neg = sim_mlp(query_neg)
        concat_sim = tf.keras.layers.concatenate([query_pos, query_neg])
        print("[SIM] concat_sim: ", concat_sim.shape)
        return concat_sim
