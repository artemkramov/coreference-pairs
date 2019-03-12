import config.db
from models.embedding.semantic_embedding import SemanticEmbedding
import numpy as np

embedding = SemanticEmbedding()

words = ['родина', 'президент', 'лалангамена']
vec1 = embedding.word2vec('президент')
vec2 = embedding.word2vec('Крамов')
cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(cos_sim)
