from os import listdir
from os.path import isfile, join
import io
from models.parse.nlp import NLP

# Load NLP parser
nlp = NLP()

directory_test = "test"
files = [join(directory_test, filename) for filename in listdir(directory_test) if
         isfile(join(directory_test, filename))]

for filename in files:
    file = io.open(filename, mode="r", encoding="utf-8")
    raw_text = file.read()
    tokens = nlp.extract_entities(raw_text)['tokens']
    entities = {}
    for token in tokens:
        if not (token.EntityID is None):
            if not (token.EntityID in entities):
                entities[token.EntityID] = []
            entities[token.EntityID].append(token)
    print(entities)
    file.close()
