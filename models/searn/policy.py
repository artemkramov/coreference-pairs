from .mention import Mention
from typing import List
import numpy as np
from ..embedding.semantic_embedding import SemanticEmbedding
from ..embedding.scalar_embedding import ScalarEmbedding
from ..sieve.sieve import Sieve
from .action import PassAction, MergeAction


# Class which defines network policy
class Policy:
    # Neural network model (binary classifier)
    model = None

    # Embedding to extract semantic data
    semantic_embedding: SemanticEmbedding = None

    # Embedding to extract scalar data
    scalar_embedding: ScalarEmbedding = None

    # Sieve to filter obvious pairs of mentions
    sieve: Sieve = None

    # Threshold to decide if clusters are compatible to be merged
    PROB_THRESHOLD = 0.5

    def __init__(self, _model):
        self.model = _model
        self.load_embeddings()

    # Load embedding models
    def load_embeddings(self):
        self.semantic_embedding = SemanticEmbedding()
        self.scalar_embedding = ScalarEmbedding()

        # Load sieve
        self.sieve = Sieve()

    # Evaluate ability for cluster merging
    def apply(self, cluster_mention: List[Mention], cluster_antecedent: List[Mention]):

        matrices = self.clusters_to_matrices(cluster_mention, cluster_antecedent)

        # Run neural network to predict
        prediction = self.model.predict_on_batch([matrices['semantic'], matrices['scalar']])[0][0]
        if prediction > self.PROB_THRESHOLD:
            return True
        return False

    # Transform pair of clusters to common matrix
    def clusters_to_matrices(self, cluster_mention: List[Mention], cluster_antecedent: List[Mention]):
        # Form pair "mention"-"antecedent"
        pair = [cluster_mention, cluster_antecedent]

        # Form matrix with all combination of entities from mention and antecedent
        pair_matrix = self.get_matrix_from_pair(pair)

        # Prepare scalar and semantic matrices
        scalar_matrix = self.get_scalar_matrix_from_pair(pair_matrix)
        semantic_matrix = self.get_semantic_matrix_from_pair(pair_matrix)
        semantic_matrix = np.expand_dims(semantic_matrix, axis=2)
        semantic_matrix = np.expand_dims(semantic_matrix, axis=0)
        scalar_matrix = np.expand_dims(scalar_matrix, axis=2)
        scalar_matrix = np.expand_dims(scalar_matrix, axis=0)
        return {
            'semantic': semantic_matrix,
            'scalar': scalar_matrix
        }

    # Preprocess document to fit features of embeddings
    def preprocess_document(self, mentions: List[Mention]):
        tokens = []
        for mention in mentions:
            tokens.extend(mention.tokens)
        self.scalar_embedding.evaluate_tfidf(tokens)
        self.semantic_embedding.tokens = tokens

    # Form matrix of pairs of entities
    # from the entity links
    @staticmethod
    def get_matrix_from_pair(pair: List[List[Mention]]):

        # Init matrix
        pair_matrix = []

        # Loop through each pair and create all possible pair variants
        # Fetch each entity data by the link
        for first_mention in pair[0]:
            for second_mention in pair[1]:
                pair_matrix.append((first_mention, second_mention))
        return pair_matrix

    def get_scalar_matrix_from_pair(self, pair_matrix):
        return self.scalar_embedding.matrix2vec(pair_matrix)

    # Get semantic matrix from the given pair of entities
    def get_semantic_matrix_from_pair(self, pair_matrix):
        matrix = []

        # From the matrix of tokens
        # form the corresponding matrix of words
        for entity_pair in pair_matrix:

            # Words of each pair
            words = []
            for entity_tokens in entity_pair:

                # Words of each entity
                entity_words = []
                for token in entity_tokens.tokens:
                    entity_words.append(token)
                words.append(entity_words)
            matrix.append(words)

        return self.semantic_embedding.matrix2vec(matrix)


class ReferencePolicy:

    @staticmethod
    def apply(state: 'State', clusters_gold):
        if (state.current_mention_idx < len(
                clusters_gold)) and state.current_mention_idx > state.current_antecedent_idx:
            if clusters_gold[state.current_mention_idx] == clusters_gold[state.current_antecedent_idx]:
                return MergeAction()
        return PassAction()
