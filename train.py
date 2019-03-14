import pickle
import configparser
from keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras import regularizers
from keras import optimizers
from keras.utils import plot_model
import keras
import numpy as np
from random import shuffle


# Class to perform compiling and training of model
class Training:
    # Model to be trained
    model = None

    # Configuration object
    config = []

    X_train = []
    Y_train = []

    def __init__(self):
        # Load configuration data
        config = configparser.ConfigParser()
        config.read('config/train.ini')
        self.config = config

    # def load_train_data(self):
    #     filename = 'dataset/data-0.pkl'
    #     handle = open(filename, 'rb')
    #     items = pickle.load(handle)
    #
    #     batches = {}
    #
    #     # Group items by the length of columns
    #     for x in items:
    #         variable_length = x['scalar'].shape[1]
    #         if not (variable_length in batches):
    #             batches[variable_length] = []
    #         batches[variable_length].append(x)
    #
    #     # Shuffle batches
    #     for key in batches:
    #         shuffle(batches[key])
    #
    #     x_sem = [np.expand_dims(x['semantic'], axis=2) for x in items]
    #     x_scal = [np.expand_dims(x['scalar'], axis=2) for x in items]
    #     self.X_train = [x_sem, x_scal]
    #     self.Y_train = [x['label'] for x in items]
    #     handle.close()

    # Split list into chunks of the fixed length
    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def train_model(self):
        config_training = self.config['TRAINING']
        filename = 'dataset/data-0.pkl'
        handle = open(filename, 'rb')
        items = pickle.load(handle)
        handle.close()

        batches = {}

        # Group items by the length of columns
        for x in items:
            variable_length = x['scalar'].shape[1]
            if not (variable_length in batches):
                batches[variable_length] = []
            x['semantic'] = np.expand_dims(x['semantic'], axis=2)
            x['scalar'] = np.expand_dims(x['scalar'], axis=2)
            batches[variable_length].append(x)

        # Split batches to mini-batches
        mini_batches = []
        for key in batches:
            # Split batches to mini-batches
            shuffle(batches[key])
            chunks = self.chunks(batches[key], int(config_training['batch_size']))
            mini_batches.extend(chunks)

        shuffle(mini_batches)

        epochs = int(config_training['epochs'])
        for epoch in range(0, epochs):

            print('Epoch {}/{}'.format(epoch, epochs))

            epoch_loss = []

            for mini_batch in mini_batches:
                x_semantic = [x['semantic'] for x in mini_batch]
                x_scalar = [x['scalar'] for x in mini_batch]
                y = [x['label'] for x in mini_batch]
                epoch_loss.append(self.model.train_on_batch([x_semantic, x_scalar], y))
            print(epoch_loss)

        # config_training = self.config['TRAINING']
        # model_history = self.model.fit(self.X_train, [self.Y_train], shuffle=True,
        #                                batch_size=int(config_training['batch_size']),
        #                                epochs=int(config_training['epochs']),
        #                                validation_split=float(config_training['validation_split']),
        #                                verbose=int(config_training['verbose']))

    # Build CNN subnetwork with 2 layers
    @staticmethod
    def build_cnn(dimension, filter_size):
        input_matrix = Input(shape=(dimension, None, 1))
        cnn = Conv2D(filter_size, activation='relu', kernel_size=(dimension, 1), data_format='channels_last')(
            input_matrix)
        cnn = GlobalMaxPooling2D(data_format='channels_last')(cnn)
        return cnn, input_matrix

    # Build model due to the configuration parameters
    def build_model(self):
        config_network = self.config['NETWORK']
        cnn_embedding, input_semantic = self.build_cnn(int(config_network['semantic_dimension']),
                                                       int(config_network['filter_size']))
        cnn_scalar, input_scalar = self.build_cnn(int(config_network['scalar_dimension']),
                                                  int(config_network['filter_size']))
        classifier = keras.layers.concatenate([cnn_embedding, cnn_scalar])
        classifier = Dense(int(config_network['hidden_layer_units']), activation="relu")(classifier)

        # Add configurable dropout layer
        classifier = Dropout(float(config_network['dropout_rate']))(classifier)

        # Load regularizer function to avoid overfitting
        kernel_regularizer_value = float(config_network['kernel_regularizer_value'])
        kernel_regularizer = None
        if config_network['kernel_regularizer'] == 'l1':
            kernel_regularizer = regularizers.l1(kernel_regularizer_value)
        if config_network['kernel_regularizer'] == 'l2':
            kernel_regularizer = regularizers.l2(kernel_regularizer_value)

        classifier = Dense(1, activation="sigmoid", kernel_regularizer=kernel_regularizer)(classifier)
        model = Model(inputs=[input_semantic, input_scalar], outputs=[classifier])

        # Load optimization function
        optimizer = optimizers.Adam()
        if config_network['optimizer'] == 'sgd':
            optimizer = optimizers.SGD(lr=float(config_network['sgd_lr']),
                                       momentum=float(config_network['sgd_momentum']),
                                       decay=float(config_network['sgd_decay']),
                                       nesterov=bool(int(config_network['sgd_nesterov'])))
        model.compile(optimizer=optimizer, loss=config_network['loss'], metrics=['accuracy'])
        plot_model(model, to_file='debug/model.png', show_shapes=True)
        self.model = model


if __name__ == "__main__":
    training = Training()
    # training.load_train_data()
    training.build_model()
    training.train_model()
