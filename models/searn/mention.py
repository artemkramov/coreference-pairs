from ..db.word import DbWord
from typing import List
from copy import copy


# Class to describe mention
class Mention:

    # Tokens of the entity
    tokens: List[DbWord] = []

    # ID of cluster
    cluster_id: str = ""

    # Check if it is an entity
    is_entity: bool = False

    def __init__(self, _tokens: List[DbWord]):
        self.tokens = _tokens.copy()


