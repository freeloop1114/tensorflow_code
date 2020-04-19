#coding: utf-8
from __future__ import print_function
import json
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
#from absl import app, flags

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from models.text_cnn_model import *
from models.text_dnn_model import *
from models.text_cdssm_model import *
from models.text_dssm_model import *
from models.text_cdssm_mlp_similarity_model import *
from models.text_dssm_mlp_similarity_model import *
from data import text_pair_data_reader 
from data import text_point_data_reader 

#FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_string('config_file', '', 'model config file')
#tf.flags.DEFINE_string('operation', 'train', 'operation type')


def model_factory(config):
    model_name = config['model']
    if 'TextDssmModel' == model_name:
        return TextDssmModel(config)
    elif 'TextCnnModel' == model_name:
        return TextCnnModel(config)
    elif 'TextDnnModel' == model_name:
        return TextDnnModel(config)
    elif 'TextCdssmModel' == model_name:
        return TextCdssmModel(config)
    elif 'TextCdssmMlpSimilarityModel' == model_name:
        return TextCdssmMlpSimilarityModel(config)
    elif 'TextDssmMlpSimilarityModel' == model_name:
        return TextDssmMlpSimilarityModel(config)
    else:
        return None


def data_factory(config):
    data_type = config['data_type']
    train_data_path = config['train_data_path']
    valid_data_path = config['valid_data_path']
    batch_size = config['batch_size']
    word_count = config['word_count']
    epochs = config['epochs']
    if 'point' == data_type:
        train_ds = text_point_data_reader.local_data(train_data_path, batch_size, epochs, word_count)
        valid_ds = text_point_data_reader.local_data(valid_data_path, batch_size, epochs, word_count)
    elif 'pair' == data_type:
        train_ds = text_pair_data_reader.local_data(train_data_path, batch_size, epochs, word_count)
        valid_ds = text_pair_data_reader.local_data(valid_data_path, batch_size, epochs, word_count)
    else:
        train_ds = None
        valid_ds = None
    return train_ds, valid_ds
    

def set_tf_config():
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    session = tf.compat.v1.Session(config = session_config)
    tf.compat.v1.keras.backend.set_session(session)
    #session_config.log_device_placement = True
    #config = tf.estimator.RunConfig(
    #        save_checkpoints_steps = 2000,
    #        keep_checkpoint_max = 10, 
    #        session_config = session_config)
    #return config


def model_config(config_file):
    with open(config_file, 'r') as f:
        data = f.read().replace('\n', '')
        config = json.loads(data)
        return config


def train(config_file):
    set_tf_config()
    config = model_config(config_file)
    model_name = config['model']
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = model_factory(config)
        if not model:
            print('[%s] not found' % (model_name))
            return 
    try:
        train_ds, valid_ds = data_factory(config)
        model.train(train_ds, valid_ds)
        #result = model.valid_step(valid_ds, 1)
        #print(result)
    except Exception as e:
        print('[%s] finish training ... ' % (model_name))
        raise e
                

def main():
    train(sys.argv[1])
    #train('./config/cnn_config.json')
    #train('./config/dnn_config.json')
    #train('./config/dssm_config.json')
    #train('./config/cdssm_config.json')
    # 1. read config
    #config = model_config('./config/dssm_config.json')
    # 2. read dataset 
    #train_data_path = '../data/'
    #valid_data_path = '../data/'
    #train_ds = text_pair_data_reader.local_train_data(train_data_path, 1)
    #valid_ds = text_pair_data_reader.local_valid_data(valid_data_path, 1)
    #train_ds = text_point_data_reader.local_train_data(train_data_path, 1)
    #valid_ds = text_point_data_reader.local_valid_data(valid_data_path, 1)

    # 3. train or predict
    #config_file = ""
    #dssm = TextCnnModel(config)
    #dssm.train_step(train_ds)
    #dssm.valid_step(valid_ds)


    #estimator = tf.estimator.Estimator(
    #        model_fn = model_fn, 
    #        model_dir = "tf_models/only_text", 
    #        config = tf_config())


if __name__ == '__main__':
    main()

