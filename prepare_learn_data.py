import config.db as db
from models.db.word import DbWord
import uuid
import itertools
from models.embedding.scalar_embedding import ScalarEmbedding
from models.embedding.semantic_embedding import SemanticEmbedding
import pickle
from random import shuffle


# Class to prepare learned data
class PrepareLearnData:
    documents = {}

    all_tokens = {}

    MAX_CLUSTER_SIZE = 7

    MAX_ENTITIES_SEPARATE_SIZE = 50

    semantic_embedding = None

    scalar_embedding = None

    # Load embedding models
    def load_embeddings(self):
        self.semantic_embedding = SemanticEmbedding()
        self.scalar_embedding = ScalarEmbedding()

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
                documents[token.DocumentID] = {'tokens': [], 'entities': {}, 'clusters': {}, 'entities_separate': [] }
            documents[token.DocumentID]['tokens'].append(token)

        # Group tokens of documents by entities
        for document_id in documents:

            entities = {}
            clusters = {}
            entities_separate = []
            document_tokens = documents[document_id]['tokens']
            for token in document_tokens:

                # Create entity group
                if token.EntityID is None:
                    entity_id = str(uuid.uuid4())
                    entities[entity_id] = [token]
                else:
                    if not (token.EntityID in entities):
                        entities[token.EntityID] = []
                    entities[token.EntityID].append(token)
                    if token.CoreferenceGroupID is None and (not (token.EntityID in entities_separate)):
                        entities_separate.append(token.EntityID)

                # Create cluster groups
                if not (token.CoreferenceGroupID is None):
                    if not (token.CoreferenceGroupID in clusters):
                        clusters[token.CoreferenceGroupID] = []
                        if not (token.EntityID in entities_separate):
                            entities_separate.append(token.EntityID)
                    if not (token.EntityID in clusters[token.CoreferenceGroupID]):
                        clusters[token.CoreferenceGroupID].append(token.EntityID)

            documents[document_id]['entities'] = entities
            documents[document_id]['clusters'] = clusters
            entities_separate.reverse()
            if len(entities_separate) > self.MAX_ENTITIES_SEPARATE_SIZE:
                entities_separate = entities_separate[:self.MAX_ENTITIES_SEPARATE_SIZE]
            documents[document_id]['entities_separate'] = entities_separate

        self.documents = documents

        # Close DB session
        db_session.close()

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

        prev_comb = None
        all_combinations = []

        while is_in_range:

            # Generate all combinations C(n=len(items), k=i)
            comb = list(itertools.combinations(items, i))

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
    def form_combinations_with_split(self, items):
        chunks = list(self.chunks(items, self.MAX_CLUSTER_SIZE))
        result = []
        for chunk in chunks:
            combinations = self.form_combinations(chunk)
            result.extend(combinations)
        return result

    # Save items as a pickle file
    def save_items(self, items, chunk_counter):
        shuffle(items)
        chunks = list(self.chunks(items, 30000))
        for idx, chunk in enumerate(chunks):
            file = 'dataset_1/data-{0}.pkl'.format(idx)
            handle = open(file, 'wb')
            pickle.dump(chunk, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
            print("Document number: %s, len=%s" % (idx, len(self.documents)))
            idx += 1
            self.scalar_embedding.evaluate_tfidf(self.documents[document_id]['tokens'])
            self.semantic_embedding.tokens = self.documents[document_id]['tokens']

            # Retrieve clusters
            # Loop through all clusters and form different combinations of correct pairs
            clusters = self.documents[document_id]['clusters']
            for cluster_id in clusters:
                items = clusters[cluster_id]
                document_dataset['correct'].extend(self.form_combinations_with_split(items))

            # Form all incorrect combinations
            document_dataset['incorrect'] = self.form_combinations_with_split(
                self.documents[document_id]['entities_separate'])

            # Loop through different labeled datasets
            class_labels = {'correct': 1, 'incorrect': 0}

            # Set the similar width of correct and incorrect labels
            if len(document_dataset['incorrect']) > len(document_dataset['correct']):
                document_dataset['incorrect'] = document_dataset['incorrect'][:len(document_dataset['correct'])]

            for label in document_dataset:
                pairs = document_dataset[label]
                for pair in pairs:
                    # Get matrix of pairs
                    pair_matrix = self.get_matrix_from_pair_links(pair, self.documents[document_id]['entities'])
                    scalar_matrix = self.get_scalar_matrix_from_pair(pair_matrix)
                    semantic_matrix = self.get_semantic_matrix_from_pair(pair_matrix)
                    item = {'label': class_labels[label], 'semantic': semantic_matrix, 'scalar': scalar_matrix}
                    data_items.append(item)
                    if len(data_items) > chunk_size:
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
    model.load_embeddings()
    model.form_train_pairs()
