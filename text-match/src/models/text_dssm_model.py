#coding: utf-8
import sys
sys.path.append("..")

import numpy as np
from keras.layers import *
from keras.models import *
import tensorflow as tf

from layers.word_embedding import *

class TextDssmModel:
    def __init__(self, model_config):
        self.neg_count = 4
        self.word_size = 128
        self.word_embedding_size = 128
        self.mlp_layers = [128, 128]
        self.model = None
        self.batch_size = 64
        self.epochs = 10

    def create_model(self):
        # 1. input data
        #query = Input(shape = (self.word_size, ), sparse = True, dtype = 'int64')
        #pos_doc = Input(shape = (self.word_size, ), sparse = True, dtype = 'int64')
        #neg_docs = [Input(shape = (self.word_size, ), sparse = True, dtype = 'int64') for i in range(self.neg_count)]
        query = Input(shape = (self.word_size, ), sparse = True)
        pos_doc = Input(shape = (self.word_size, ), sparse = True)
        neg_docs = [Input(shape = (self.word_size, ), sparse = True) for i in range(self.neg_count)]
        print("query: ", query.shape)

        word_embedding = Dense(self.word_embedding_size, kernel_initializer = 'random_normal')
        query_embedding = word_embedding(tf.keras.backend.to_dense(query))
        pos_doc_embedding = word_embedding(tf.keras.backend.to_dense(pos_doc))
        neg_docs_embedding = [word_embedding(tf.keras.backend.to_dense(neg_doc)) for neg_doc in neg_docs]
        #query_embedding = sparse_text_embedding(query, [self.word_size, self.word_embedding_size])
        #tf.compat.v1.get_variable_scope().reuse_variables()
        #pos_doc_embedding = sparse_text_embedding(pos_doc, [self.word_size, self.word_embedding_size])
        #neg_docs_embedding = [sparse_text_embedding(neg_doc, [self.word_size, self.word_embedding_size]) for neg_doc in neg_docs]

        # 2. mlp_layer
        for i in range(len(self.mlp_layers)):
            query_mlp = Dense(self.mlp_layers[i], activation = 'relu', kernel_initializer = 'random_normal')
            doc_mlp = Dense(self.mlp_layers[i], activation = 'relu', kernel_initializer = 'random_normal')
            query_embedding = query_mlp(query_embedding)
            print("query_embedding: ", query_embedding.shape)
            pos_doc_embedding = doc_mlp(pos_doc_embedding)
            print("pos_doc_embedding: ", pos_doc_embedding.shape)
            neg_docs_embedding = [doc_mlp(neg_doc) for neg_doc in neg_docs_embedding]

        # 3. sim
        pos_doc_sim = dot([query_embedding, pos_doc_embedding], axes = 1, normalize = True)
        neg_doc_sims = [dot([query_embedding, neg_doc], axes = 1, normalize = True) for neg_doc in neg_docs]
        print("pos_doc_sim: ", pos_doc_sim.shape)

        # 4. loss
        concat_sim = concatenate([pos_doc_sim] + neg_doc_sims)
        print("concat_sim: ", concat_sim.shape)
        prob = Activation("softmax")(concat_sim)

        self.model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
        self.model.compile(
            optimizer = "adadelta",
            loss = "categorical_crossentropy",
            metrics = [tf.keras.metrics.AUC()])

    def train(self, data, label):
        self.model.fit(data, label, batch_size = self.batch_size, epochs = self.epochs, verbose = 2)

# test
if __name__ == '__main__':
    # create test data
    sample_size = 1
    query_data = []
    pos_doc_data = []
    neg_doc_data = [[] for i in range(4)]
    for i in range(sample_size):
        query = []
        for j in range(128):
            v = np.random.randint(0, 127)
            query.append(v)
        query_data.append(query)

        pos_doc = []
        for j in range(128):
            v = np.random.randint(0, 127)
            pos_doc.append(v)
        pos_doc_data.append(pos_doc)

    for i in range(sample_size):
        for j in range(4):
            #while True:
            #    negative = np.random.randint(0, sample_size)
            #    if negative != j:
            #        break
            negative = 0
            neg_doc_data[j].append(pos_doc_data[negative])

    labels = np.zeros((sample_size, 4 + 1))
    labels[:, 0] = 1
    query_data = np.array(query_data)
    pos_doc_data = np.array(pos_doc_data)
    for j in range(4):
        neg_doc_data[j] = np.array(neg_doc_data[j])

    print(query_data)
    model = TextDssmModel(None)
    model.create_model()
    model.train([query_data, pos_doc_data] + [neg_doc_data[j] for j in range(4)], labels)