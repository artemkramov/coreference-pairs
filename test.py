from os import listdir
from os.path import isfile, join
import os
import io
import configparser
from keras.models import model_from_json
from models.searn.policy import Policy, ReferencePolicy
from models.searn.mention import Mention
from models.searn.agent import Agent
from typing import List
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
import joblib
import dill
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier


class Test:

    # Folder to save models during training
    folder_models = "bin"

    # Template to save models
    filename_template = "model_{0}"

    folder_conll = "test/conll"

    folder_texts = "test/texts"

    model = None

    model_semantic = None

    epoch = '-'

    transformers = None

    def __init__(self):
        # Load configuration data
        config = configparser.ConfigParser()
        config.read('config/train.ini')
        self.config = config

    # Build model due to the configuration parameters
    def build_model_semantic(self, epoch_number=None):
        embedding_size = 1024 * 4
        lstm_units = 128
        dense_units = 128

        model = CoreferentClusterModel(embedding_size, lstm_units, dense_units)
        if not (epoch_number is None):
            subfolder = join(self.folder_models, str(epoch_number))
            # Prepare filenames of the model
            filename_weights = self.filename_template.format(epoch_number)
            model.load_weights(join(subfolder, filename_weights))
            print("Model is loaded")
            self.epoch = epoch_number
        self.model_semantic = model

    # Build model due to the configuration parameters
    def build_model(self):
        self.model = joblib.load(join(self.folder_models, "extra_trees_clf_1.pkl"))
        self.transformers = dill.load(open(join(self.folder_models, "transformers_1.pickle"), mode='rb'))

    def load_model(self, epoch_number):
        self.epoch = epoch_number
        subfolder = join(self.folder_models, str(epoch_number))
        # Prepare filenames of the model
        filename = self.filename_template.format(epoch_number)
        filename_json = join(subfolder, filename + ".json")
        filename_weights = join(subfolder, filename + ".h5")
        json_file = open(filename_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(filename_weights)
        print("Model is loaded")
        self.model = model

    def save_file(self, conll, document_id, is_actual=True):
        extension = "key"
        if not is_actual:
            extension = "response"
        filename = "{0}.{1}".format(document_id, extension)
        filename = join(self.folder_conll, filename)

        handle = open(filename, mode='w', encoding='utf-8')
        handle.write(conll)
        handle.close()

    def save_file_texts(self, lines, epoch):
        extension = "txt"
        filename = "{0}.{1}".format(epoch, extension)
        filename = join(self.folder_texts, filename)

        handle = open(filename, mode='w', encoding='utf-8')
        handle.write("\r\n".join(lines))
        handle.close()

    def run(self):

        # Load mentions from DB
        folder = 'dataset_2'
        config_training = self.config['TRAINING']
        files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

        # Policy to learn
        policy = Policy(self.model, self.transformers)
        reference_policy = ReferencePolicy()

        policy.model_semantic = self.model_semantic

        # Percentage of documents for training purpose
        training_split = float(config_training['training_split'])

        for filename in files:
            # Read documents
            handle = open(filename, 'rb')
            documents: List[List[Mention]] = pickle.load(handle)
            # documents = documents[:1]
            handle.close()

            # Calculate separator index to divide documents into 2 parts
            separator_index = int(training_split * len(documents)) + 1

            predict = []
            actual = []

            # separator_index = 2300

            print(len(documents))

            lines = []

            predictions = []

            for document_id, document in enumerate(documents[separator_index:]):
                print(document_id)
                agent = Agent(document)
                agent.set_gold_state(document)
                #agent.set_sieve()
                policy.preprocess_document(document)
                agent.move_to_end_state(policy)
                #print(agent.predictions)
                #predictions = agent.predictions

                conll_predict = agent.state_to_conll(agent.states[-1], document_id)
                conll_actual = agent.state_to_conll(agent.state_gold, document_id)
                predict.append(conll_predict)
                actual.append(conll_actual)

                words_predicted, groups_predicted = agent.state_to_list(agent.states[-1], document_id)
                words_actual, groups_actual = agent.state_to_list(agent.state_gold, document_id)

                lines.append("Actual: " + " ".join(words_actual))
                lines.append("Coreferent pairs: " + str(groups_actual))
                lines.append("Predicted: " + " ".join(words_predicted))
                lines.append("Coreferent pairs: " + str(groups_predicted))
                lines.append("\r\n")
                # self.save_file(conll_predict, document_id, False)
                # self.save_file(conll_actual, document_id, True)
                #print(agent.actions)'''

            file = self.epoch
            self.save_file(os.linesep.join(predict), file, False)
            self.save_file(os.linesep.join(actual), file, True)
            self.save_file_texts(lines, self.epoch)
            # with open('test/predictions_1.pkl', 'wb') as file:
            #     dill.dump(predictions, file)


class CoreferentClusterModel(Model):

  def __init__(self, embedding_size, lstm_units, dense_units, **kwargs):
    super(CoreferentClusterModel, self).__init__(**kwargs)

    self.bilstm = Bidirectional(LSTM(lstm_units, activation='tanh', recurrent_activation="sigmoid", input_shape=(None, embedding_size), dtype=tf.float32))
    self.dense1 = Dense(dense_units)
    self.dense2 = Dense(1, activation='sigmoid')

  def call(self, inputs, training=None, mask=None):

    cluster1 = self.bilstm(inputs[0])
    cluster2 = self.bilstm(inputs[1])

    cluster = tf.concat([cluster1, cluster2], axis=-1)
    x = self.dense1(cluster)

    return self.dense2(x)


# Get list of files to examine
if __name__ == "__main__":
    test = Test()
    test.build_model()
    test.build_model_semantic(20)
    test.run()
