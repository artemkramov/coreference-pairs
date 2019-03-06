import numpy as np
import random
import io


# Class for performing of semantic transformations (word-vec, text-vec etc.)
class SemanticEmbedding:

    # Model which contains pairs word-vector
    model = {}

    # Keys of model
    model_keys = []

    # File path to load model
    model_path = "models/news.lowercased.lemmatized.glove.300d"

    # Constructor of the model
    def __init__(self):
        self.load_dictionary()

    # Load pre-trained model
    def load_dictionary(self):
        print('Loading semantic model...')
        f = io.open(self.model_path, mode='r', encoding='utf-8')

        # Loop through each line and parse it
        # First token is word, another tokens form the corresponding vector
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(value) for value in split_line[1:]])
            self.model[word] = embedding
            self.model_keys.append(word)
        f.close()

    # Convert any word to vector
    def word2vec(self, word):
        if word in self.model:
            return self.model[word]
        random_word = random.choice(self.model_keys)
        return self.model[random_word]
