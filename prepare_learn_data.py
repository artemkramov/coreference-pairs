import config.db as db
from models.db.word import DbWord
import uuid
import itertools


# Class to prepare learned data
class PrepareLearnData:

    documents = {}

    MAX_CLUSTER_SIZE = 7

    MAX_ENTITIES_SEPARATE_SIZE = 50

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

    # Form different combinations
    def form_train_pairs(self):
        dataset = {'correct': [], 'incorrect': []}
        for document_id in self.documents:
            clusters = self.documents[document_id]['clusters']
            for cluster_id in clusters:
                items = clusters[cluster_id]
                dataset['correct'].extend(self.form_combinations_with_split(items))
            dataset['incorrect'] = self.form_combinations_with_split(self.documents[document_id]['entities_separate'])
        a = 4


if __name__ == "__main__":
    model = PrepareLearnData()
    model.load_documents()
    model.form_train_pairs()
