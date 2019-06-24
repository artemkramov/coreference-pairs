from os import listdir
from os.path import isfile, join
import io
from models.cluster_model import CoreferenceModel
import configparser
from keras.models import model_from_json
from models.searn.policy import Policy
from models.searn.mention import Mention
from models.searn.agent import Agent
from typing import List
import pickle


class Test:

    # Folder to save models during training
    folder_models = "bin"

    # Template to save models
    filename_template = "model_{0}"

    model = None

    def __init__(self):
        # Load configuration data
        config = configparser.ConfigParser()
        config.read('config/train.ini')
        self.config = config

    def load_model(self, epoch_number):
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

    def run(self):

        # Load mentions from DB
        folder = 'dataset_2'
        config_training = self.config['TRAINING']
        files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

        # Policy to learn
        policy = Policy(self.model)

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

            for document_id, document in enumerate(documents[separator_index:]):
                agent = Agent(document)
                agent.set_gold_state(document)
                policy.preprocess_document(document)
                agent.move_to_end_state(policy)
                #agent.state_to_conll(agent.states[-1], document_id)
                agent.state_to_conll(agent.state_gold, document_id)
                break


# Get list of files to examine
if __name__ == "__main__":
    test = Test()
    test.load_model(1)
    test.run()
