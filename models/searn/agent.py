from .mention import Mention
from typing import List
from .policy import Policy, ReferencePolicy
from .action import MergeAction, PassAction, Action
from .state import State
import uuid
import copy
import time
import os


class Agent:

    # List of states
    states: List[State] = []

    # All tokens of the document
    tokens = []

    # List of mentions
    mentions = []

    # Gold state
    state_gold: State = None

    # From input matrices to compute
    @staticmethod
    def get_matrices(policy, state_current, mentions):

        # Get clusters of the mention and antecedent
        cluster_mention = state_current.get_cluster_of_mention(state_current.current_mention_idx, mentions)
        cluster_antecedent = state_current.get_cluster_of_mention(state_current.current_antecedent_idx, mentions)

        return policy.clusters_to_matrices(cluster_mention, cluster_antecedent)

    def state_to_conll(self, state, document_id, offset=0):
        document_id = str(document_id)
        header = ["#begin document ({0});".format(document_id), "part 000"]
        lines = [' '.join(header)]
        entity_counter = offset
        groups = {}
        c = []
        counter = 0
        number = 0
        for cluster_id in state.clusters:
            if len(cluster_id) > 0:
                if not (cluster_id in groups):
                    groups[cluster_id] = counter
                    number = counter
                    counter += 1
                else:
                    number = groups[cluster_id]
            else:
                number = counter
                counter += 1
            c.append(number)

        for mention_id, mention in enumerate(self.tokens):
            line = [document_id, '-']
            for idx, token in enumerate(mention.tokens):
                line_format = '-'
                if mention.is_entity:
                    if idx == 0:
                        entity_counter += 1
                    if len(mention.tokens) == 1:
                        line_format = '({0})'
                    else:
                        if idx == 0:
                            line_format = '({0}'
                        if idx == len(mention.tokens) - 1:
                            line_format = '{0})'

                    line[1] = line_format.format(c[entity_counter - 1])
                lines.append(' '.join(line))
        pass

    def form_training_set(self, policy, actions, policy_reference, metric):

        counter = 0

        training_set = []

        # clusters = []
        # for m in self.state_gold.clusters:
        #     clusters.append(m)
        # print(len(list(dict.fromkeys(clusters))))
        #
        # for m in self.states[0].clusters:
        #     clusters.append(m)
        # print(len(list(dict.fromkeys(clusters))))

        # Evaluate loss function for all state in trajectory besides the last state
        for state in self.states[:-1]:
            losses = []
            #print("State from trajectory: {0}, len={1}".format(counter, len(self.states[:-1])))
            counter += 1

            # Apply each action to the current state
            for action in actions:

                agent_inner = Agent(self.tokens, False)
                agent_inner.states.append(state)

                # Apply current action (perform one step)
                state_one_step = agent_inner.move(policy, action)
                # print("After one step")

                start = time.clock()

                # If we can't move more than make deep copy of the state
                # Else continue to move with the reference policy till the end state
                if not state_one_step:
                    state_end = copy.deepcopy(state)
                else:
                    agent_inner.states.append(state_one_step)
                    state_end = agent_inner.move_to_end_state(policy_reference, self.state_gold)[-1]

                #print("Time to count action: {0}".format(time.clock() - start))

                # clusters = []
                # for m in state_end.clusters:
                #     clusters.append(m)
                # print(len(list(dict.fromkeys(clusters))))

                # Evaluate loss function by computing corresponding metric between gold state
                #  and actual end state
                losses.append(1 - metric.evaluate(state_end, self.state_gold))

            # Append state with corresponding losses to the training set
            training_set.append({
                'losses': losses,
                'state': state,
                'mentions': self.mentions,
                'tokens': self.tokens
            })

        return training_set

    # Performing the necessary action depending on the current situation
    def move(self, policy: Policy, action: Action = None):

        state_current = self.get_current_state()

        # If it is the end of the list or there is just a single mention
        # than return corresponding flag
        if state_current.current_mention_idx == len(self.mentions) or len(self.mentions) < 2:
            return False

        if action is None:

            # Get clusters of the mention and antecedent
            cluster_mention = state_current.get_cluster_of_mention(state_current.current_mention_idx, self.mentions)
            cluster_antecedent = state_current.get_cluster_of_mention(state_current.current_antecedent_idx, self.mentions)

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

        state_new = action.run(state_current)

        # Reset current mention and antecedent for new state
        if is_merge:
            state_new.current_antecedent_idx = state_new.current_mention_idx
            state_new.current_mention_idx += 1
        else:
            if state_new.current_antecedent_idx > 0:
                state_new.current_antecedent_idx -= 1
            else:
                state_new.current_antecedent_idx = state_new.current_mention_idx
                state_new.current_mention_idx += 1
        return state_new

    # Modify state till the end
    def move_to_end_state(self, policy, state_gold: 'State' = None):

        # Define trajectory which contains all states
        state_init = self.get_current_state()
        next_step_exists = True
        state_current = state_init
        c = 1
        import time
        start = time.clock()

        # While it is possible to perform 'move' operation
        while next_step_exists:
            c += 1
            # If the 'gold' state isn't defined
            # than move with learn policy
            # Else use reference policy to move
            if state_gold is None:
                action = None
            else:
                action = policy.apply(state_current, state_gold.clusters)

            # Make move
            state_new = self.move(policy, action)

            # If it is no more possible to move again
            # than stop process
            # Else continue with new state
            if not state_new:
                next_step_exists = False
            else:
                self.states.append(state_new)
                state_current = state_new
        #print("Iterations: {0}, {1}".format(c, time.clock() - start))
        return self.states

    def push_state(self, clusters):
        state = State(clusters)
        self.states.append(state)

    def get_current_state(self):
        return self.states[-1]

    def set_gold_state(self, tokens):
        clusters = []
        for mention in tokens:
            if mention.is_entity:
                clusters.append(mention.cluster_id)
        self.state_gold = State(clusters)

    def __init__(self, _tokens: List[Mention], start=True):
        self.states = []

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
            clusters = [str(uuid.uuid4()) for mention in self.mentions]

            # Add start state
            self.push_state(clusters)

    def set_tokens(self, _tokens):
        self.tokens = copy.deepcopy(_tokens)
        self.mentions = []
