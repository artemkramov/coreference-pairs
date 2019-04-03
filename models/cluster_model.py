from .parse.nlp import NLP
from .sieve.sieve import Sieve
import itertools
import os
import sys
import inspect
import numpy as np
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from os.path import join
from keras.models import model_from_json
from prepare_learn_data import PrepareLearnData


# Class to perform cluster division of the text
class CoreferenceModel:
    # NLP model to make operations with raw text
    nlp = None

    # Model to make sieve filtering
    sieve = None

    # Neural network model
    model_neural = None

    # Model to prepare data for neural network
    model_prepare = None

    # Entities of the current document
    entities = {}

    # Init function
    def __init__(self):

        # Load neural network
        directory = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'bin')
        alias_model = 'model'
        f = open(join(directory, alias_model + '.json'), 'r')
        self.model_neural = model_from_json(f.read())
        self.model_neural.load_weights(join(directory, alias_model + '.h5'))
        f.close()

        # Load all necessary models
        self.nlp = NLP()
        self.sieve = Sieve()
        self.model_prepare = PrepareLearnData()
        self.model_prepare.load_embeddings()

    # Process clusters by the given current index and corresponding antecedent position
    def process_clusters(self, clusters, pairs_sieve):

        # Init start indices
        current_index = 1
        antecedent_index = 0

        while True:
            if len(clusters) <= current_index:
                break

            # Run algorithm to find out if clusters can be merged
            is_merge_required = self.is_possible_to_merge(clusters[current_index], clusters[antecedent_index],
                                                          pairs_sieve)

            if is_merge_required:
                # Merge clusters to antecedent position
                clusters[antecedent_index].extend(clusters[current_index])

                # Remove current cluster
                del clusters[current_index]

            # If the position of the antecedent is the first at the document
            # than change current cluster
            if antecedent_index == 0:
                # Increment current position
                # If the merge has been performed
                # than set next position as the current one
                next_index = current_index + 1
                if is_merge_required:
                    next_index -= 1
                current_index = next_index
                antecedent_index = next_index - 1
            else:
                # Decrease antecedent position
                # If merge has been performed
                # than set current position as the antecedent position
                if is_merge_required:
                    current_index = antecedent_index
                antecedent_index -= 1
        return clusters

    # Check if clusters should be merged
    def is_possible_to_merge(self, cluster1, cluster2, pairs_sieve):

        # Sieve result for this pair of clusters
        result_sieve = None

        pair_entity = []

        # Firstly try to check sieve results to verify if either clusters can be merged or merge operation is denied
        # Check all entities of clusters entity by entity
        for entity1 in cluster1:
            if result_sieve is not None:
                break
            for entity2 in cluster2:
                entity1_id = entity1[0].EntityID
                entity2_id = entity2[0].EntityID

                # Loop through pair sieves
                for pair in pairs_sieve:
                    if (pair[0] == entity1_id and pair[1] == entity2_id) or (
                            pair[0] == entity2_id and pair[1] == entity1_id):
                        result_sieve = pair[2]
                        break

        # If sieve result contains definite result about this entities
        # than return result
        if result_sieve is not None:
            return result_sieve

        # Form pairs
        pair_entity.append(tuple([entity[0].EntityID for entity in cluster1]))
        pair_entity.append(tuple([entity[0].EntityID for entity in cluster2]))

        pair_matrix = self.model_prepare.get_matrix_from_pair_links(pair_entity, self.entities)
        scalar_matrix = self.model_prepare.get_scalar_matrix_from_pair(pair_matrix)
        semantic_matrix = self.model_prepare.get_semantic_matrix_from_pair(pair_matrix)
        semantic_matrix = np.expand_dims(semantic_matrix, axis=2)
        semantic_matrix = np.expand_dims(semantic_matrix, axis=0)
        scalar_matrix = np.expand_dims(scalar_matrix, axis=2)
        scalar_matrix = np.expand_dims(scalar_matrix, axis=0)

        prediction = self.model_neural.predict([semantic_matrix, scalar_matrix])
        if prediction[0][0] > 0.98:
            return True

        return False

    # Find coreference pairs from tokens given
    def find_coreference_pairs_from_tokens(self, tokens):

        # Preprocess tokens
        self.model_prepare.scalar_embedding.evaluate_tfidf(tokens)
        self.model_prepare.semantic_embedding.tokens = tokens

        # Find direct speech groups
        direct_speech_groups = self.sieve.find_direct_speech(tokens)

        # Group tokens by entities
        entities = {}
        for token in tokens:
            if not (token.EntityID is None):
                if not (token.EntityID in entities):
                    entities[token.EntityID] = []
                entities[token.EntityID].append(token)

        # Find aliases
        aliases = self.sieve.find_aliases(entities)

        # Set current entities
        self.entities = entities

        # Form paired combinations of entities to apply sieve
        combinations = list(itertools.combinations(entities.keys(), 2))

        # Init pairs of sieve
        pairs_sieve = []

        # Loop through combinations list
        for comb in combinations:
            first_entity = entities[comb[0]]
            second_entity = entities[comb[1]]

            # Apply sieve to the pair of entities
            result = self.sieve.apply([first_entity, second_entity], direct_speech_groups, tokens, aliases)

            # If the sieve decision is definite (True/False)
            # Than save that pair iof entities
            if result is not None:
                pairs_sieve.append((comb[0], comb[1], result))

        # Init basic set of clusters
        clusters = [[entities[entity]] for entity in entities]
        if len(clusters) > 1:
            clusters = self.process_clusters(clusters, pairs_sieve)
            clusters = [cluster for cluster in clusters if len(cluster) > 1]
            self.print_clusters(clusters)
        pass

    @staticmethod
    def print_clusters(clusters):
        if len(clusters) == 0:
            print("No clusters")
        for idx, cluster in enumerate(clusters):
            print("Cluster with idx %s" % idx)
            for entity in cluster:
                words = []
                for token in entity:
                    words.append(token.RawText)
                print(' '.join(words))
            print("============================")

    # Find co-reference pairs inside the text
    def find_coreference_pairs_from_text(self, text):

        # Form tokens
        tokens = self.nlp.extract_entities(text)['tokens']

        # Find coreference pairs from tokens
        self.find_coreference_pairs_from_tokens(tokens)
