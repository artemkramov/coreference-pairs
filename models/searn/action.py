import copy


# Action to run
class Action:

    # Label of the action
    name = ''

    # Function to apply
    def run(self, state: 'State') -> 'State':
        pass


# Merge action
class MergeAction(Action):

    name = 'merge'

    def run(self, state: 'State') -> 'State':

        # Init new state with deep copy
        state_new = copy.deepcopy(state)
        # Get current mention and its antecedent and merge them into one cluster
        state_new.clusters[state_new.current_mention_idx] = state_new.clusters[state_new.current_antecedent_idx]

        return state_new


# Pass action
class PassAction(Action):

    name = 'pass'

    def run(self, state: 'State') -> 'State':
        # Just copy state without any action
        return copy.deepcopy(state)

