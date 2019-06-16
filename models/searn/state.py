from .mention import Mention
import uuid
from typing import List
from .action import MergeAction, PassAction, Action
from .policy import Policy
import copy


# State of mentions
class State:

    # Index of the current mention in the list of mentions
    current_mention_idx = 1

    # Index of the current antecedent in the list of mentions
    current_antecedent_idx = 0

    clusters = []

    def __init__(self, _clusters):
        self.clusters = _clusters

    def get_cluster_id(self, mention_id):
        return self.clusters[mention_id]

    # Get cluster with mentions where the given mention is located
    def get_cluster_by_id(self, mention_cluster_id: str, mentions: List[Mention]) -> List[Mention]:
        if mention_cluster_id is None:
            return []
        cluster = []
        for idx, cluster_id in enumerate(self.clusters):
            if cluster_id == mention_cluster_id:
                cluster.append(mentions[idx])
        return cluster

    def get_cluster_of_mention(self, mention_id, mentions):
        mention_cluster_id = self.get_cluster_id(mention_id)
        return self.get_cluster_by_id(mention_cluster_id, mentions)



