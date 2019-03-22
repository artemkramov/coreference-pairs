import numpy as np
import random
import io
from models.embedding.base_embedding import BaseEmbedding
from models.dictionary import PartOfSpeech


# Class for performing of semantic transformations (word-vec, text-vec etc.)
class SemanticEmbedding(BaseEmbedding):
    # Model which contains pairs word-vector
    model = {}

    # Keys of model
    model_keys = []

    # File path to load model
    model_path = "models/vectors.50d"

    # Tokens of the current document
    tokens = []

    # Constant which denotes the number of words used to get vector representation of the context
    OFFSET = 5

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
        if word.lower() in self.model:
            return self.model[word.lower()]
        random_word = random.choice(self.model_keys)
        return self.model[random_word]

    # Get vector representation of the next 5 words
    def next2vec(self, token):
        next_tokens = []

        # If it's the last token than duplicate it
        if token.WordOrder == len(self.tokens) - 1:
            next_tokens.append(token)
        else:
            i = token.WordOrder + 1
            # Loop through array till the start of tokens
            while len(next_tokens) < self.OFFSET:
                # Stop if it is the start of array
                if i == len(self.tokens):
                    break
                current_token = self.tokens[i]
                # Check if current token isn't a punctuation symbol
                if current_token.PartOfSpeech != PartOfSpeech.PUNCT:
                    next_tokens.append(current_token)
                i += 1
        if len(next_tokens) == 0:
            next_tokens.append(token)
        return self.entity2vec(next_tokens)

    # Get vector representation of the 5 previous words
    def prev2vec(self, token):
        prev_tokens = []

        # If it's the first word than just duplicate it
        if token.WordOrder == 0:
            prev_tokens.append(token)
        else:
            i = token.WordOrder - 1
            # Loop through array till the start of tokens
            while len(prev_tokens) < self.OFFSET:
                # Stop if it is the start of array
                if i < 0:
                    break
                current_token = self.tokens[i]
                # Check if current token isn't a punctuation symbol
                if current_token.PartOfSpeech != PartOfSpeech.PUNCT:
                    prev_tokens.append(current_token)
                i -= 1
        if len(prev_tokens) == 0:
            prev_tokens.append(token)
        return self.entity2vec(prev_tokens)

    # Convert pair of entities to a single vector
    def pair2vec(self, pair_words):
        matrix = []
        for words in pair_words:
            matrix.append(self.entity2vec(words))
        vector_first = self.entity2vec(pair_words[0])
        vector_second = self.entity2vec(pair_words[1])
        vector_prev_first = self.prev2vec(pair_words[0][0])
        vector_next_first = self.next2vec(pair_words[0][-1])
        vector_prev_second = self.prev2vec(pair_words[1][0])
        vector_next_second = self.next2vec(pair_words[1][-1])
        return np.concatenate(
            (vector_first, vector_second, vector_prev_first, vector_next_first, vector_prev_second, vector_next_second), axis=None)

    # Convert a set of words (entity) to a single vector
    def entity2vec(self, words):
        matrix = []
        for word in words:
            matrix.append(self.word2vec(word.Lemmatized))
        return np.mean(np.array(matrix), axis=0)
