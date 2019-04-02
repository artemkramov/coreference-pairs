from .parse.nlp import NLP
from .sieve.sieve import Sieve
import itertools


# Class to perform cluster division of the text
class CoreferenceModel:

    # NLP model to make operations with raw text
    nlp = None

    # Model to make sieve filtering
    sieve = None

    def __init__(self):
        # Load all necessary models
        self.nlp = NLP()
        self.sieve = Sieve()

    # Find co-reference pairs inside the text
    def find_coreference_pairs(self, text):

        # Form tokens
        tokens = self.nlp.extract_entities(text)['tokens']

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

        pass
