import numpy as np


# General class for both embeddings: semantic and scalar
class BaseEmbedding:

    # Convert set of pairs to a matrix representation
    # Don't forget to transpose matrix in the end
    def matrix2vec(self, pair_matrix):
        vectors = []
        for pair_words in pair_matrix:
            vectors.append(self.pair2vec(pair_words))
        return np.array(vectors).transpose()

    # Mock for the common function of all children
    def pair2vec(self, pair_words):
        raise NotImplementedError()
