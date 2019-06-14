from .mention import Mention
import uuid
from typing import List
from .action import MergeAction, PassAction, Action
from .policy import Policy
import copy


# State of mentions
class State:

    # All tokens of the document
    tokens = []

    # Index of the current mention in the list of mentions
    current_mention_idx = 2

    # Index of the current antecedent in the list of mentions
    current_antecedent_idx = 1

    # List of mentions
    mentions = []

    # From input matrices to compute
    def get_matrices(self, policy):
        # Get current mention
        mention = self.mentions[self.current_mention_idx]

        # Get current antecedent
        antecedent = self.mentions[self.current_antecedent_idx]

        # Get clusters of the mention and antecedent
        cluster_mention = self.get_cluster_of_mention(mention.cluster_id)
        cluster_antecedent = self.get_cluster_of_mention(antecedent.cluster_id)

        return policy.clusters_to_matrices(cluster_mention, cluster_antecedent)

    # Performing the necessary action depending on the current situation
    def move(self, policy: Policy, action: Action = None):

        # If it is the end of the list or there is just a single mention
        # than return corresponding flag
        if self.current_mention_idx == len(self.mentions) or len(self.mentions) < 2:
            return False

        if action is None:
            # Get current mention
            mention = self.mentions[self.current_mention_idx]

            # Get current antecedent
            antecedent = self.mentions[self.current_antecedent_idx]

            # Get clusters of the mention and antecedent
            cluster_mention = self.get_cluster_of_mention(mention.cluster_id)
            cluster_antecedent = self.get_cluster_of_mention(antecedent.cluster_id)

            # Apply policy to find out which action should be applied
            is_merge = policy.apply(cluster_mention, cluster_antecedent)

            # Run action
            action = PassAction()
            if is_merge:
                action = MergeAction()
        else:
            is_merge = False
            if isinstance(action, MergeAction):
                is_merge = True

        state_new = action.run(self)

        # Reset current mention and antecedent for new state
        if is_merge:
            state_new.current_antecedent_idx = state_new.current_mention_idx
            state_new.current_mention_idx += 1
        else:
            if state_new.current_antecedent_idx > 1:
                state_new.current_antecedent_idx -= 1
            else:
                state_new.current_antecedent_idx = state_new.current_mention_idx
                state_new.current_mention_idx += 1
        return state_new

    # Get cluster with mentions where the given mention is located
    def get_cluster_of_mention(self, mention_cluster_id: str) -> List[Mention]:
        if mention_cluster_id is None:
            return []
        return [mention for mention in self.mentions if mention.cluster_id == mention_cluster_id]

    # Modify state till the end
    def move_to_end_state(self, policy, state_gold: 'State' = None):

        # Define trajectory which contains all states
        trajectory = [self]
        next_step_exists = True
        state_current = self

        # While it is possible to perform 'move' operation
        while next_step_exists:

            # If the 'gold' state isn't defined
            # than move with learn policy
            # Else use reference policy to move
            if state_gold is None:
                action = None
            else:
                action = policy.apply(self, state_gold)
            # Make move
            state_new = state_current.move(policy, action)

            # If it is no more possible to move again
            # than stop process
            # Else continue with new state
            if not state_new:
                next_step_exists = False
            else:
                trajectory.append(state_new)
                state_current = state_new
        return trajectory

    def __init__(self, _tokens: List[Mention], start=True):
        # Set all tokens of the document
        self.set_tokens(_tokens)

        # Extract all entities
        for mention in self.tokens:
            if mention.is_entity:
                self.mentions.append(mention)

        # If start flag is set
        # than make all mentions as a singleton clusters
        if start:
            # Set each mention as a singleton cluster
            for mention in self.mentions:
                mention.cluster_id = str(uuid.uuid4())

    def set_tokens(self, _tokens):
        # self.tokens = []
        # self.mentions = []
        self.tokens = copy.deepcopy(_tokens)
        self.mentions = []
        # for token in _tokens:
        #     mentions = copy.deepcopy(token.tokens)
        #     mention = Mention(mentions)
        #     mention.cluster_id = token.cluster_id
        #     mention.is_entity = token.is_entity
        #     self.tokens.append(mention)


