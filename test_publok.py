from os import listdir
from os.path import isfile, join
import os
import io
import configparser
from keras.models import model_from_json
from models.searn.policy import Policy
from models.searn.mention import Mention
from models.searn.agent import Agent
from models.db.word import DbWordPublok, DbWord
from typing import List
import pickle
import config.db_publok as db


class Test:

    # Folder to save models during training
    folder_models = "bin"

    # Template to save models
    filename_template = "model_{0}"

    folder_conll = "test/conll"

    model = None

    epoch = '-'

    def __init__(self):
        # Load configuration data
        config = configparser.ConfigParser()
        config.read('config/train.ini')
        self.config = config

        # Database session marker
        self.db_session = db.session()

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

    def run(self):

        # Load tokens from DB and group by texts
        tokens: List[DbWordPublok] = self.db_session.query(DbWordPublok).all()
        texts = {}
        for token in tokens:
            if not (token.DocumentID in texts):
                texts[token.DocumentID] = []
            texts[token.DocumentID].append(token)

        # Loop through each text
        documents = []
        group_ids = []
        for text_id in texts:

            documents.append([])

            # Group all text words by entity ids
            groups = []
            for token in texts[text_id]:
                attributes = dict(token.__dict__)
                del attributes['WordID']
                del attributes['_sa_instance_state']
                token_db = DbWord()
                token_db.__dict__ = attributes

                if token_db.EntityID is None:
                    groups.append([token_db])
                else:
                    if not (token_db.EntityID in group_ids):
                        group_ids.append(token_db.EntityID)
                        groups.append([token_db])
                    else:
                        groups[-1].append(token_db)

            # Create mentions and add them to documents
            for words in groups:
                mention = Mention(words)
                documents[-1].append(mention)

        # Policy to learn
        policy = Policy(self.model)

        for document_id, document in enumerate(documents):
            agent = Agent(document)
            agent.set_sieve()
            policy.preprocess_document(document)
            agent.move_to_end_state(policy)
            if len(agent.states[0].clusters) > 0:
                print("Good")
            conll_predict = agent.state_to_conll(agent.states[-1], document_id)
            pass


# Get list of files to examine
if __name__ == "__main__":
    test = Test()
    test.load_model(9)
    test.run()
