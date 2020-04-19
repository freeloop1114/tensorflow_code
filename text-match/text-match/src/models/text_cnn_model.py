#coding: utf-8
import sys
sys.path.append("..")

import numpy as np
import scipy as sp
import tensorflow as tf
from models.base_model import BaseModel

class TextCnnModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)

    def build_graph(self):
        # 1. input data
        query_input = tf.keras.Input(shape = (self.word_count, ), dtype = tf.int64, name = "query")
        pos_input = tf.keras.Input(shape = (self.word_count, ), dtype = tf.int64, name = "pos_data")
        masking = tf.keras.layers.Masking(mask_value = 0)
        query_mask = masking(query_input)
        pos_mask = masking(pos_input)

        # 2. word embedding
        query, pos_doc = self.word_embedding(query_mask, pos_mask)
        query, pos_doc = self.cnn1d(query, pos_doc)
        query_pos = self.mlp(query, pos_doc)
        pos_doc_sim = self.similarity(query_pos)

        # 6. classification
        prob = tf.keras.layers.Activation("sigmoid")(pos_doc_sim)
        print("prob: ", prob.shape)

        # 7. set loss and optimizer
        self.model = tf.keras.Model(inputs = [query_input, pos_input], outputs = [prob])
        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(
            optimizer = optimizer,
            loss = tf.keras.losses.binary_crossentropy,
            metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()]
        )
        print(self.model.summary())


    def word_embedding(self, query_input, pos_input):
        word = tf.keras.layers.Embedding(
            input_dim = self.word_size,
            output_dim = self.word_embedding_size,
            embeddings_initializer = 'glorot_uniform',
        )

        dropout = tf.keras.layers.Dropout(self.dropout)
        query = dropout(word(query_input))
        pos_doc = dropout(word(pos_input))
        #query = tf.keras.backend.sum(word(query_input), axis = 1)
        #pos_doc = tf.keras.backend.sum(word(pos_input), axis = 1)
        print("query: ", query.shape)
        return query, pos_doc

    
    def cnn1d(self, query, pos_doc):
        for i, layer in enumerate(self.cnn_layers):
            query_outputs = []
            pos_doc_outputs = []
            for conv in layer:
                # query conv and pooling
                query_conv = tf.keras.layers.Conv1D(
                        conv[0], conv[1], conv[2], 
                        activation = tf.nn.relu
                    )
                query_conv_embedding = query_conv(query)
                query_pooling = tf.keras.layers.MaxPool1D(
                        pool_size = int(query_conv_embedding.shape[1])
                    )
                query_pooling_embedding = query_pooling(query_conv_embedding)
                query_outputs.append(query_pooling_embedding)

                # pos_doc conv and pooling
                pos_doc_conv = tf.keras.layers.Conv1D(
                        conv[0], conv[1], conv[2], 
                        activation = tf.nn.relu
                    )
                pos_doc_conv_embedding = pos_doc_conv(pos_doc)
                pos_doc_pooling = tf.keras.layers.MaxPool1D(
                        pool_size = int(pos_doc_conv_embedding.shape[1])
                    )
                pos_doc_pooling_embedding = pos_doc_pooling(pos_doc_conv_embedding)
                pos_doc_outputs.append(pos_doc_pooling_embedding)
            # concat every conv which has diff filter size
            query = tf.concat(query_outputs, 2)  
            pos_doc = tf.concat(pos_doc_outputs, 2)  
            print("[CNN_%d] query: " % (i), query.shape)

        # reshape to [?, X]
        query = tf.reshape(query, [-1, query.shape[-1]])
        pos_doc = tf.reshape(pos_doc, [-1, pos_doc.shape[-1]])
        return query, pos_doc


    def mlp(self, query, pos_doc):
        query_pos = tf.concat([query, pos_doc], axis=-1)
        for i, output_dim in enumerate(self.mlp_layers):
            layer = tf.keras.layers.Dense(
                            output_dim,
                            activation = 'relu', 
                            kernel_initializer = 'glorot_normal'
                        )
            query_pos = layer(query_pos)
            print("[MLP_%d] query_pos: " % (i), query_pos.shape)
        return query_pos 


    def similarity(self, query_pos):
        sim_mlp = tf.keras.layers.Dense(1)
        concat_sim = sim_mlp(query_pos)
        print("[SIM] concat_sim: ", concat_sim.shape)
        return concat_sim

