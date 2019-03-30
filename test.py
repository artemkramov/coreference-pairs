from os import listdir
from os.path import isfile, join
import io
from models.sieve.sieve import Sieve
import pickle
from models.parse.nlp import NLP
import itertools

# Load NLP parser
nlp = NLP()

# Init sieve handler
sieve = Sieve()

directory_test = "test"
files = [join(directory_test, filename) for filename in listdir(directory_test) if
         isfile(join(directory_test, filename))]

for filename in files:
    file = io.open(filename, mode="r", encoding="utf-8")
    raw_text = file.read()
    f = open("tmp/tokens.pkl", 'rb')
    tokens = nlp.extract_entities(raw_text)['tokens']
    f.close()
    entities = {}
    direct_speech_groups = sieve.find_direct_speech(tokens)

    # for token in tokens:
    #     if token.WordOrder in direct_speech_groups:
    #         print(direct_speech_groups[token.WordOrder]['group_id'], direct_speech_groups[token.WordOrder]['role'], token.RawText)

    for token in tokens:
        if not (token.EntityID is None):
            if not (token.EntityID in entities):
                entities[token.EntityID] = []
            entities[token.EntityID].append(token)
    combinations = list(itertools.combinations(entities.keys(), 2))
    for comb in combinations:
        first_entity = entities[comb[0]]
        second_entity = entities[comb[1]]
        result = sieve.apply([first_entity, second_entity], direct_speech_groups, tokens)
        if result is not None:
            words = []
            for entity in first_entity:
                words.append(entity.RawText)
            words.append("|")
            for entity in second_entity:
                words.append(entity.RawText)
            print("Result: %s, %s" % (result, ' '.join(words)))

    file.close()
