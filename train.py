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
import os
from os import listdir
from os.path import isfile, join
import shutil
from typing import List
from models.searn.mention import Mention
from models.searn.state import State
from models.searn.policy import Policy, ReferencePolicy
from models.searn.metric import BCubed
from models.searn.action import MergeAction, PassAction
import copy
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import keras.backend as K
import time

# keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


# Class to perform compiling and training of model
class Training:
    # Model to be trained
    model = None

    # Configuration object
    config = []

    # Template to save models
    filename_template = "model_{0}"

    # Folder to save models during training
    folder_models = "bin"

    train_losses = K.variable([435])

    def __init__(self):
        # Load configuration data
        config = configparser.ConfigParser()
        config.read('config/train.ini')
        self.config = config

    # Split list into chunks of the fixed length
    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # Train model
    def train_model(self):
        folder = 'dataset'
        config_training = self.config['TRAINING']
        files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

        epochs = int(config_training['epochs'])
        model_history = {
            'loss': [],
            'acc': [],
            'val_loss': [],
            'val_acc': []
        }
        # The percentage of validation set
        validation_split = float(config_training['validation_split'])

        # Number of epochs to check the overfitting
        patience = int(config_training['patience'])

        for epoch in range(0, epochs):

            print('Epoch {}/{}'.format(epoch, epochs))

            # Init counters to calculate metrics
            epoch_loss_total = 0
            epoch_acc_total = 0
            epoch_val_loss_total = 0
            epoch_val_acc_total = 0
            batch_counter = 0
            batch_val_counter = 0

            # Loop through each dataset file and load it every epoch
            # These files are too big to load in memory therefore they are loaded during every epoch
            for filename in files:
                handle = open(filename, 'rb')
                items = pickle.load(handle)
                handle.close()

                batches = {}
                print("Filename %s" % filename)

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

                # Split mini-batches into training and validation sets
                separator_index = len(mini_batches) - int(validation_split * len(mini_batches)) - 1
                mini_batches_train = mini_batches[:separator_index]
                mini_batches_validation = mini_batches[separator_index:]

                # Process training mini-batches
                for mini_batch in mini_batches_train:
                    metrics = self.process_mini_batch(mini_batch, 'train')
                    if len(metrics) > 0:
                        epoch_loss_total += metrics[0]
                        epoch_acc_total += metrics[1]
                        batch_counter += 1

                # Calculate metrics for the validation set
                for mini_batch in mini_batches_validation:
                    metrics = self.process_mini_batch(mini_batch, 'test')
                    if len(metrics) > 0:
                        epoch_val_loss_total += metrics[0]
                        epoch_val_acc_total += metrics[1]
                        batch_val_counter += 1

            # Count all metrics as average value per the whole epoch
            epoch_loss = epoch_loss_total / batch_counter
            epoch_acc = epoch_acc_total / batch_counter
            epoch_val_loss = epoch_val_loss_total / batch_val_counter
            epoch_val_acc = epoch_val_acc_total / batch_val_counter

            # Write counted metrics to the model history object
            model_history['loss'].append(epoch_loss)
            model_history['acc'].append(epoch_acc)
            model_history['val_loss'].append(epoch_val_loss)
            model_history['val_acc'].append(epoch_val_acc)
            print("loss: {0}, acc: {1}, val_loss: {2}, val_acc: {3}".format(epoch_loss, epoch_acc, epoch_val_loss,
                                                                            epoch_val_acc))
            # Check if the check of early stopping should be performed
            if epoch + 1 > patience:
                prev_loss = model_history['val_loss'][-patience]
                if prev_loss < epoch_val_loss:
                    print("Early stopping detected, stop the training process...")
                    break
                else:
                    self.save_model(epoch, True)
            else:
                self.save_model(epoch)

        # Leave just the best variant of model
        self.leave_best_model(model_history['val_loss'], patience)

    # Leave just the best one model
    def leave_best_model(self, loss_history, patience):
        i = len(loss_history) - 1
        best_idx = i

        # Find the index from the loss history with the lowest val_loss score
        while i > len(loss_history) - 1 - patience:
            if loss_history[i] < loss_history[best_idx]:
                best_idx = i
            i -= 1

        # Loop through sorted list of subfolders
        # And leave folder with list index equal to the best index
        subfolders = os.listdir(self.folder_models)
        subfolders.sort()
        for idx, folder in enumerate(subfolders):
            if i == idx:
                continue
            shutil.rmtree(join(self.folder_models, folder))

    # Save current model to the filesystem
    def save_model(self, epoch, shift=False):

        # Create subfolder
        subfolder = join(self.folder_models, str(epoch))
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        # Prepare filenames of the model
        filename = self.filename_template.format(epoch)
        filename_json = join(subfolder, filename + ".json")
        filename_weights = join(subfolder, filename + ".h5")

        # Save configuration of the model
        model_json = self.model.to_json()
        handle = open(filename_json, 'w')
        handle.write(model_json)
        handle.close()

        # Save weights
        self.model.save_weights(filename_weights)

        # Check if it is necessary to remove the most outdated model
        if shift:
            # Get list of folders which contain saved models per epoch
            subfolders = os.listdir(self.folder_models)
            subfolders.sort()

            # Remove the most outdated version
            shutil.rmtree(join(self.folder_models, subfolders[0]))

    # Process mini-batch corresponding to the given mode
    def process_mini_batch(self, mini_batch, mode='train'):
        # Form semantic and scalar matrices in separate way
        # Also generate output vectors
        x_semantic = [x['semantic'] for x in mini_batch]
        x_scalar = [x['scalar'] for x in mini_batch]
        y = [x['id'] for x in mini_batch]
        y = np.array(y)

        # If it is the training mini-batch than pass batch with changing of weights
        # Else just pass mini-batch and retrieve metrics

        if mode == 'train':
            metrics = self.model.train_on_batch([x_semantic, x_scalar], y)
        else:
            metrics = self.model.test_on_batch([x_semantic, x_scalar], y)

        return metrics

    # Build CNN subnetwork with 2 layers
    @staticmethod
    def build_cnn(dimension, filter_size):
        input_matrix = Input(shape=(dimension, None, 1))
        cnn = Conv2D(filter_size, activation='relu', kernel_size=(dimension, 1), data_format='channels_last')(
            input_matrix)
        cnn = GlobalMaxPooling2D(data_format='channels_last')(cnn)
        return cnn, input_matrix

    def loss_function(self, y_true, y_pred):

        y_true_int = tf.cast(y_true, dtype=tf.int32)
        # y_true_int = K.print_tensor(y_true_int, message="y_true_int")
        losses_current = tf.gather(self.train_losses, y_true_int)
        self.train_losses = K.print_tensor(self.train_losses, message="losses_current")
        losses_current = tf.squeeze(losses_current, [1])
        y_pred_negative = tf.map_fn(lambda x: (1 - x), y_pred, dtype=tf.float32)
        y_pred_concat = tf.concat([y_pred, y_pred_negative], 1)
        t = K.sum(y_pred_concat * losses_current)
        t = K.print_tensor(t, message="t: ")

        return K.mean(K.sum(y_pred_concat * losses_current, axis=1))
        # return tf.convert_to_tensor(np.array([[1.0]]))
        # return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

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

        self.model = model

    def recompile_model(self):
        config_network = self.config['NETWORK']
        # Load optimization function
        optimizer = optimizers.Adam()
        if config_network['optimizer'] == 'sgd':
            optimizer = optimizers.SGD(lr=float(config_network['sgd_lr']),
                                       momentum=float(config_network['sgd_momentum']),
                                       decay=float(config_network['sgd_decay']),
                                       nesterov=bool(int(config_network['sgd_nesterov'])))
        loss_function = tf.contrib.eager.defun(self.loss_function)
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=[])
        plot_model(self.model, to_file='debug/model.png', show_shapes=True)
        print(self.model.summary())

    def train_searn(self):

        # Load mentions from DB
        folder = 'dataset_2'
        config_training = self.config['TRAINING']
        files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

        # Policy to learn
        policy = Policy(self.model)

        # Reference policy
        policy_reference = ReferencePolicy()

        # Metric to evaluate
        metric = BCubed()

        # Possible actions
        actions = [MergeAction(), PassAction()]

        # Get total number of epochs
        epochs = int(config_training['epochs'])

        # Percentage of documents for training purpose
        training_split = float(config_training['training_split'])

        # Loop through each epoch
        for epoch in range(0, epochs):
            print('Epoch {}/{}'.format(epoch, epochs))

            for filename in files:

                # Read documents
                handle = open(filename, 'rb')
                documents: List[List[Mention]] = pickle.load(handle)
                # documents = documents[:1]
                handle.close()

                training_set = []

                # Calculate separator index to divide documents into 2 parts
                separator_index = len(documents) - int(training_split * len(documents)) - 1
                # print(separator_index)

                for document_id, document in enumerate(documents[:separator_index]):

                    start = time.clock()

                    # Set initial state and end state
                    state_initial = State(document)
                    state_last_gold = State(document, False)

                    clusters = []
                    for m in state_last_gold.mentions:
                        clusters.append(m.cluster_id)
                    print(len(list(dict.fromkeys(clusters))))

                    print("Process document {0} from {1}".format(document_id, separator_index))

                    policy.preprocess_document(document)
                    trajectory = state_initial.move_to_end_state(policy)
                    counter = 1

                    # Evaluate loss function for all state in trajectory besides the last state
                    for state in trajectory[:-1]:
                        losses = []
                        print("State from trajectory: {0}, len={1}".format(counter, len(trajectory)))
                        counter += 1

                        # Apply each action to the current state
                        for action in actions:

                            # Apply current action (perform one step)
                            state_one_step = state.move(policy, action)

                            # If we can't move more than make deep copy of the state
                            # Else continue to move with the reference policy till the end state
                            if not state_one_step:
                                state_end = copy.deepcopy(state)
                            else:
                                state_end = state_one_step.move_to_end_state(policy_reference, state_last_gold)[-1]

                            # Evaluate loss function by computing corresponding metric between gold state
                            #  and actual end state
                            losses.append(metric.evaluate(state_end, state_last_gold))

                        # Append state with corresponding losses to the training set
                        training_set.append({
                            'losses': losses,
                            'state': state
                        })

                    print("Time for document: {0}".format(time.clock() - start))

                # Prepare list of losses to collect losses for training
                losses_train = []

                # Dictionary of batches
                # We will group batches by the size
                batches = {}

                start = time.clock()
                print("Run training process, examples: {0}".format(len(training_set)))

                # Loop through collected training set
                for idx, training_example in enumerate(training_set):

                    # Fetch losses and corresponding state
                    expected = training_example['losses']
                    state = training_example['state']

                    # Preprocess documents with the policy to evaluate embeddings
                    policy.preprocess_document(state.tokens)

                    # Get matrix of the state and remove extra dimension
                    # which corresponds to the length of the batch
                    x = state.get_matrices(policy)
                    x['semantic'] = np.squeeze(x['semantic'], axis=0)
                    x['scalar'] = np.squeeze(x['scalar'], axis=0)

                    # Set order number of the example
                    # It will be used to match the state with computed losses after a shuffle
                    x['id'] = idx
                    losses_train.append(expected)

                    # Group examples by the length
                    variable_length = x['scalar'].shape[1]
                    if not (variable_length in batches):
                        batches[variable_length] = []
                    batches[variable_length].append(x)

                # Convert list of training losses to a tensor model
                # and recompile model with changed train losses
                self.train_losses = tf.constant(losses_train)
                self.recompile_model()

                # Split batches to mini-batches
                mini_batches = []
                for key in batches:
                    # Split batches to mini-batches
                    shuffle(batches[key])
                    chunks = self.chunks(batches[key], int(config_training['batch_size']))
                    mini_batches.extend(chunks)

                shuffle(mini_batches)

                # Process training mini-batches
                for mini_batch in mini_batches:
                    metrics = self.process_mini_batch(mini_batch, 'train')
                    print("Metrics: {0}".format(metrics))

                # Save model
                self.save_model(epoch)

                print("Time for training: {0}".format(time.clock() - start))


if __name__ == "__main__":

    training = Training()
    # training.load_train_data()
    training.build_model()
    # training.train_model()
    training.train_searn()
