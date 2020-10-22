import numpy as np
import random
import io
from models.embedding.base_embedding import BaseEmbedding
from models.dictionary import PartOfSpeech
from elmoformanylangs import Embedder
from os.path import join
from scipy.spatial.distance import cosine


FOLDER = ''


# Class for performing of semantic transformations (word-vec, text-vec etc.)
class SemanticEmbedding(BaseEmbedding):
    model = None

    # File path to load model
    model_path = "elmo"

    # Tokens of the current document
    tokens = []

    # Constant which denotes the number of words used to get vector representation of the context
    OFFSET = 5

    phrase_cache = {}

    # Constructor of the model
    def __init__(self):
        self.load_dictionary()

    # Load pre-trained model
    def load_dictionary(self):
        print('Loading semantic model...')
        self.model = Embedder(join(FOLDER, self.model_path))

    def clear_phrase_cache(self):
        self.phrase_cache = {}

    def get_pair_similarity(self, pair):
        phrase1 = [token.RawText for token in pair[0].tokens]
        phrase2 = [token.RawText for token in pair[1].tokens]

        all_tokens = [phrase1, phrase2]

        embeddings = self.model.sents2elmo(all_tokens)
        embeddings = [np.mean(embedding, axis=0) for embedding in embeddings]

        # print(embeddings[0].shape)

        return cosine(embeddings[0], embeddings[1])

    def cluster_to_matrix(self, cluster):
        phrases = [[token.RawText for token in mention.tokens] for mention in cluster]
        vectors = []
        for mention in cluster:
            phrase = [token.RawText for token in mention.tokens]
            prev_tokens = self.prev_words(mention.tokens[0])
            prev_tokens.reverse()
            next_tokens = self.next_words(mention.tokens[-1])
            sentence_tokens = self.sentence_words(mention.tokens[0])

            all_tokens = [prev_tokens, phrase, next_tokens, sentence_tokens]

            key = "".join(["".join(sentence) for sentence in all_tokens])
            if not (key in self.phrase_cache):
                try:
                    embeddings = self.model.sents2elmo(all_tokens)
                    embeddings = [np.mean(embedding, axis=0) for embedding in embeddings]
                    # print(embeddings[0].shape)
                except ZeroDivisionError:
                    print(all_tokens)
                vector = np.concatenate(embeddings)
                self.phrase_cache[key] = vector
            else:
                vector = self.phrase_cache[key]
            vectors.append(vector)

        # print(len(vectors))
        # print(vectors[0].shape)
        matrix = np.asarray(vectors)
        return matrix

    def sentence_words(self, token):
        i = token.WordOrder - 1
        words = []
        while i >= 0:

            if self.tokens[i].RawTagString == './SENT_END':
                break

            words.append(self.tokens[i].RawText)

            i -= 1

        words.reverse()

        i = token.WordOrder
        while i < len(self.tokens):

            if self.tokens[i].RawTagString == './SENT_END':
                break

            words.append(self.tokens[i].RawText)

            i += 1

        return words

    # Get vector representation of the 5 previous words
    def prev_words(self, token):
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
                try:
                    current_token = self.tokens[i]
                except IndexError:
                    pass
                # Check if current token isn't a punctuation symbol
                if current_token.PartOfSpeech != PartOfSpeech.PUNCT:
                    prev_tokens.append(current_token)
                i -= 1
        if len(prev_tokens) == 0:
            prev_tokens.append(token)
        return [word.RawText for word in prev_tokens]

    # Get vector representation of the next 5 words
    def next_words(self, token):
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
        return [word.RawText for word in next_tokens]