#coding: utf-8
import tensorflow as tf

class BaseModel:
    def __init__(self, config):
        # network config
        self.neg_count = config['neg_count']
        self.word_count = config['word_count']
        self.word_size = config['word_size']
        self.word_embedding_size = config['word_embedding_size']
        self.mlp_layers = config['mlp_layers']
        self.cnn_layers = config['cnn_layers']         
        self.dropout = config['dropout']

        # training config
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.train_steps = config['train_steps']
        self.valid_steps = config['valid_steps']
        self.verbose = 1

        # output_path
        self.name = config['name']
        self.save_path = config['save_path'] + '/' + self.name
        self.logs = config['logs'] + '/' + self.name
        self.callbacks = []
        self.model = None
        self.build_callbacks()
        self.build_graph()

    def build_callbacks(self):
        model_check_point = tf.keras.callbacks.ModelCheckpoint(
                filepath = '%s_{epoch}' % (self.save_path),
                save_best_only = True,
                monitor = 'val_loss',
                save_freq = 10000,
                verbose = 0
                )
        tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir = self.logs,
                update_freq = 100
                )
        early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=3
                )
        self.callbacks.append(model_check_point)
        self.callbacks.append(tensorboard)
        self.callbacks.append(early_stop)


    def build_graph(self):
        pass


    def train(self, train_data, valid_data):
        return self.model.fit(
            train_data,
            epochs = self.epochs, 
            steps_per_epoch = self.train_steps,
            validation_data = valid_data,
            validation_steps = self.valid_steps,
            verbose = self.verbose, 
            callbacks = self.callbacks
        )

    def evaluate(self, data):
        return self.model.evaluate(
            data,
            verbose = self.verbose, 
            steps = self.valid_steps 
        )

    def save(self):
        return self.model.save(
            self.save_path, 
            save_format = 'tf'
        )

    def predict(self, data):
        return self.model.predict(data)
