import config.db as db
from models.db.word import DbWord
from models.sieve.sieve import Sieve
import uuid
import itertools
from models.embedding.scalar_embedding import ScalarEmbedding
from models.embedding.semantic_embedding import SemanticEmbedding
import pickle
from random import shuffle
import scipy
from models.searn.mention import Mention
from typing import List
import json
import dill


class MentionWeb:

    # Tokens of the entity
    tokens = []

    # ID of cluster
    cluster_id: str = ""

    # Check if it is an entity
    is_entity: bool = False

    def __init__(self, _tokens):
        self.tokens = _tokens.copy()


class DbWordWeb:

    ID = 0
    RawText = ""
    DocumentID = ""
    WordOrder = ""
    PartOfSpeech = ""
    Lemmatized = ""
    IsPlural = ""
    IsProperName = ""
    IsHeadWord = ""
    Gender = ""
    EntityID = ""
    RawTagString = ""
    CoreferenceGroupID = ""
    RemoteIPAddress = ""


# Class to prepare learned data
class PrepareLearnData:
    documents = {}

    all_tokens = {}

    MAX_CLUSTER_SIZE = 30

    MAX_ENTITIES_SEPARATE_SIZE = 50

    semantic_embedding = None

    scalar_embedding = None

    sieve = None

    # Load embedding models
    def load_embeddings(self):
        self.semantic_embedding = SemanticEmbedding()
        self.scalar_embedding = ScalarEmbedding()

        # Load sieve
        self.sieve = Sieve()

    # Load all documents from DB and group it
    def load_documents(self):

        documents = {}

        # Database session marker
        db_session = db.session()

        # Get tokens from DB
        tokens = db_session.query(DbWord).all()

        # Group tokens by documents
        for token in tokens:
            if not (token.DocumentID in documents):
                documents[token.DocumentID] = {'tokens': [], 'entities': {}, 'clusters': {}, 'entities_separate': []}
            token_web = DbWordWeb()
            properties = token.__dict__
            properties.pop('_sa_instance_state', None)

            for key in properties:
                setattr(token_web, key, properties[key])

            documents[token.DocumentID]['tokens'].append(token_web)
            # documents[token.DocumentID]['tokens'].append(token)

        document_mentions: List[List[MentionWeb]] = []

        # Group tokens of documents by entities
        for document_id in documents:

            document_tokens = documents[document_id]['tokens']

            mentions: List[MentionWeb] = []
            counter = 0

            while counter < len(document_tokens):
                token = document_tokens[counter]

                if token.EntityID is None:
                    mention = MentionWeb([token])
                    mention.is_entity = False
                else:
                    current_entity_tokens = [token]
                    while counter + 1 < len(document_tokens) and document_tokens[counter + 1].EntityID == token.EntityID:
                        current_entity_tokens.append(document_tokens[counter + 1])
                        counter += 1
                    mention = MentionWeb(current_entity_tokens)
                    mention.is_entity = True
                    if not (token.CoreferenceGroupID is None):
                        mention.cluster_id = token.CoreferenceGroupID

                mentions.append(mention)
                counter += 1

            document_mentions.append(mentions)

        # Close DB session
        db_session.close()

        self.save_items(document_mentions)

    # Split two tuples into two tuples
    # that don't have intersection part
    @staticmethod
    def check_pair(a, b):

        # Transform the tuples to the sets
        A = set(a)
        B = set(b)

        # Return the smaller tuple and difference between them
        if len(A) > len(B):
            return b, tuple(A - B)
        else:
            return a, tuple(B - A)

    # Push pair of tuples to array
    @staticmethod
    def push_pair(a, b, items):
        # Check if pair of tuples doesn't already exist
        if (not ([a, b] in items)) and (not ([b, a] in items)):
            items.append([a, b])

    # Form different combinations from the given list
    def form_combinations(self, items):
        # Condition flag to exit from loop
        is_in_range = True

        # Start with the number 2
        i = 2

        # Maximum length of combinations
        max_comb_length = 30

        # Maximum expected length of combination
        max_expected_length = 200

        prev_comb = None
        all_combinations = []

        while is_in_range:

            if int(scipy.special.comb(len(items), i)) < max_expected_length:

                # Generate all combinations C(n=len(items), k=i)
                comb = list(itertools.combinations(items, i))
                shuffle(comb)

                # Cut comb list to omit memory overflow
                if len(comb) > max_comb_length:
                    comb = comb[:max_comb_length]

                # Loop through all elements inside generated elements
                j = 0
                while j < len(comb):

                    # Inner loop
                    # Try to merge with siblings
                    k = j
                    while k < len(comb):
                        first, second = self.check_pair(comb[k], comb[j])
                        if len(second) > 0:
                            self.push_pair(first, second, all_combinations)
                        k += 1
                    j += 1

                # If it is the first stage
                # Than just write it to result list
                if prev_comb is None:
                    for c in comb:
                        self.push_pair((c[0],), (c[1],), all_combinations)

                prev_comb = comb

            i += 1
            is_in_range = i < len(items)
        return all_combinations

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # Form combinations from items due to limits
    def form_combinations_with_split(self, items, document, direct_speech_groups):
        shuffle(items)
        chunks = list(self.chunks(items, self.MAX_CLUSTER_SIZE))
        result = []
        for chunk in chunks:
            combinations = self.form_combinations(chunk)

            # Remove all combinations that can be resolved by the sieve
            combinations_corrected = []

            for comb in combinations:
                is_combination_complex = True

                # Loop trough all tokens inside both entities
                # and apply sieve for them
                for entity1_id in comb[0]:
                    if not is_combination_complex:
                        break

                    for entity2_id in comb[1]:
                        result_sieve = self.sieve.apply(
                            [document['entities'][entity1_id], document['entities'][entity2_id]], direct_speech_groups,
                            document['tokens'], {})
                        if result_sieve is not None:
                            is_combination_complex = False
                            break

                if is_combination_complex:
                    combinations_corrected.append(comb)
            result.extend(combinations_corrected)
        return result

    # Save items as a pickle file
    def save_items(self, items, chunk_counter=0):
        shuffle(items)
        chunks = list(self.chunks(items, 30000))
        for idx, chunk in enumerate(chunks):
            file = 'dataset_3/data-web-{0}-{1}.pkl'.format(idx, chunk_counter)
            handle = open(file, 'wb')
            dill.dump(chunk, handle, protocol=dill.HIGHEST_PROTOCOL)
            handle.close()
            print('Save chunk %s, len=%s' % (chunk_counter, len(chunk)))

    # Form different combinations
    def form_train_pairs(self):
        data_items = []
        chunk_size = 3000000
        chunk_counter = 0
        idx = 0

        for document_id in self.documents:
            document_dataset = {'correct': [], 'incorrect': []}

            idx += 1

            print("Document number: %s, len=%s" % (idx, len(self.documents)))

            self.scalar_embedding.evaluate_tfidf(self.documents[document_id]['tokens'])
            self.semantic_embedding.tokens = self.documents[document_id]['tokens']

            direct_speech_groups = self.sieve.find_direct_speech(self.documents[document_id]['tokens'])

            # Retrieve clusters
            # Loop through all clusters and form different combinations of correct pairs
            clusters = self.documents[document_id]['clusters']
            for cluster_id in clusters:
                items = clusters[cluster_id]
                document_dataset['correct'].extend(
                    self.form_combinations_with_split(items, self.documents[document_id],
                                                      direct_speech_groups))

            # Form all incorrect combinations
            document_dataset['incorrect'] = self.form_combinations_with_split(
                self.documents[document_id]['entities_separate'], self.documents[document_id],
                direct_speech_groups)

            # Loop through different labeled datasets
            class_labels = {'correct': 1, 'incorrect': 0}

            # Set the similar width of correct and incorrect labels
            # if len(document_dataset['incorrect']) > 6 * len(document_dataset['correct']):
            #     document_dataset['incorrect'] = document_dataset['incorrect'][:6 * len(document_dataset['correct'])]

            for label in document_dataset:
                pairs = document_dataset[label]
                for pair in pairs:
                    # Get matrix of pairs
                    pair_matrix = self.get_matrix_from_pair_links(pair, self.documents[document_id]['entities'])
                    scalar_matrix = self.get_scalar_matrix_from_pair(pair_matrix)
                    semantic_matrix = self.get_semantic_matrix_from_pair(pair_matrix)

                    item = {'label': class_labels[label], 'semantic': semantic_matrix, 'scalar': scalar_matrix}
                    data_items.append(item)
                    if len(data_items) > 500000:
                        self.save_items(data_items, chunk_counter)
                        chunk_counter += 1
                        data_items = []
        if len(data_items) > 0:
            self.save_items(data_items, chunk_counter)

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
                for token in entity_tokens:
                    entity_words.append(token)
                words.append(entity_words)
            matrix.append(words)

        return self.semantic_embedding.matrix2vec(matrix)

    # Form matrix of pairs of entities
    # from the entity links
    @staticmethod
    def get_matrix_from_pair_links(pair, entities):

        # Init matrix
        pair_matrix = []

        # Loop through each pair and create all possible pair variants
        # Fetch each entity data by the link
        for first_entity_id in pair[0]:
            first_entity = entities[first_entity_id]
            for second_entity_id in pair[1]:
                second_entity = entities[second_entity_id]
                pair_matrix.append((first_entity, second_entity))
        return pair_matrix


if __name__ == "__main__":
    model = PrepareLearnData()
    model.load_documents()
    # model.load_embeddings()
    # model.form_train_pairs()
