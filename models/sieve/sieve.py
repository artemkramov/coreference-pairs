from .direct_speech import DirectSpeech
from abc import ABC, abstractmethod
from models.dictionary import PartOfSpeech


# Class to apply a set of sieves
class Sieve:
    # Model for working with direct speech
    model_direct_speech = None

    # Constructor
    def __init__(self):
        self.model_direct_speech = DirectSpeech()

    # Find direct speech groups among tokens
    def find_direct_speech(self, tokens):
        return self.model_direct_speech.find_direct_speech_groups(tokens)

    # Apply all handlers
    @staticmethod
    def apply(entities, direct_speech_groups, tokens):
        discourse_handler = DiscourseHandler()
        full_symbol_handler = FullSymbolHandler()
        discourse_handler.set_next(full_symbol_handler)
        return discourse_handler.handle_request(entities, direct_speech_groups, tokens)


# Common class for Chain of Responsibility
class Handler(ABC):
    """
    Define an interface for handling requests.
    Implement the successor link.
    """

    def __init__(self, successor=None):
        self._successor = successor

    def set_next(self, successor):
        self._successor = successor
        return successor

    @abstractmethod
    def handle_request(self, entities, direct_speech_groups, tokens):
        if self._successor is not None:
            return self._successor.handle_request(entities, direct_speech_groups, tokens)
        return None


# Handler to analyze direct speech
class DiscourseHandler(Handler):

    def handle_request(self, entities, direct_speech_groups, tokens):
        # Get first tokens from every entity
        token_first_entity = entities[0][0]
        token_second_entity = entities[1][0]

        # Check if tokens are located inside direct speech
        if token_first_entity.WordOrder in direct_speech_groups and token_second_entity.WordOrder in direct_speech_groups:

            token_first_group = direct_speech_groups[token_first_entity.WordOrder]
            token_second_group = direct_speech_groups[token_second_entity.WordOrder]

            if token_first_group['group_id'] == token_second_group['group_id']:

                """" First rule
                всі займенники «я», що цитуються одним автором, посилаються на один об'єкт
                всі займенники «ти», що цитуються одним автором, посилаються на один об'єкт
                """
                if token_first_group['role'] == token_second_group['role'] \
                        and token_first_entity.Lemmatized.lower() == token_second_entity.Lemmatized.lower() \
                        and token_first_entity.PartOfSpeech == PartOfSpeech.PRONOUN \
                        and token_second_entity.PartOfSpeech == PartOfSpeech.PRONOUN \
                        and len(entities[0]) == 1 and len(entities[1]) == 1 \
                        and (
                        token_first_entity.Lemmatized.lower() == 'я' or token_first_entity.Lemmatized.lower() == 'ти'):
                    return True

                """ Second rule
                автор і займенник «я» в цитатах автора посилаються на один об'єкт
                автор і всі об’єкти у його висловлюваннях, крім займенника «я», відсіюються
                """
                if token_first_group['role'] != token_second_group['role']:
                    author = entities[0]
                    pronoun = entities[1]
                    if token_second_group['role'] == 'author':
                        author = entities[1]
                        pronoun = entities[0]
                    if self.is_entity_is_single_author(author, direct_speech_groups, tokens) and len(pronoun) == 1 \
                            and pronoun[0].PartOfSpeech == PartOfSpeech.PRONOUN:
                        if pronoun[0].Lemmatized.lower() == 'я':
                            return True
                        else:
                            return False

                """ Third rule
                різні особові займенники, використані в цитатах одного автора, відсіюються
                """
                if token_first_group['role'] == token_second_group['role'] and token_first_group['role'] == 'direct':
                    if token_first_entity.PartOfSpeech == PartOfSpeech.PRONOUN \
                            and token_second_entity.PartOfSpeech == PartOfSpeech.PRONOUN \
                            and len(entities[0]) == 1 and len(entities[1]) == 1 \
                            and token_first_entity.Lemmatized.lower() != token_second_entity.Lemmatized.lower():
                        return False

                """ Fourth rule
                іменникові фрази і займенники «я», «ти», «ми» в одному висловлюванні відсіюються
                """
                if token_first_group['role'] == token_second_group['role'] and token_first_group['role'] == 'direct':
                    if token_first_entity.PartOfSpeech == PartOfSpeech.PRONOUN \
                            or token_second_entity.PartOfSpeech == PartOfSpeech.PRONOUN:
                        pronoun = entities[0]
                        np = entities[1]
                        if token_second_entity.PartOfSpeech == PartOfSpeech.PRONOUN:
                            pronoun = entities[1]
                            np = entities[0]
                        if len(pronoun) == 1 and pronoun[0].Lemmatized.lower() in ["я", "ти", "ми"] \
                                and (np[0].PartOfSpeech != PartOfSpeech.PRONOUN or len(np) > 1):
                            return False

        return super().handle_request(entities, direct_speech_groups, tokens)

    # Check if the entity is a single entity inside the author part
    @staticmethod
    def is_entity_is_single_author(entity, direct_speech_groups, tokens):
        # Get direct speech group
        entity_group = direct_speech_groups[entity[0].WordOrder]
        counter = 0

        # Loop through all speech groups
        for token_word_order in direct_speech_groups:
            # Search the token with the same group id and author part
            if direct_speech_groups[token_word_order]['group_id'] == entity_group['group_id'] and \
                    direct_speech_groups[token_word_order]['role'] == 'author':

                # If the token is inside entity group
                # then increment counter
                if tokens[token_word_order].EntityID is not None:
                    counter += 1

        # If length pf entity equals to counter
        if len(entity) == counter:
            return True
        return False


class FullSymbolHandler(Handler):

    def handle_request(self, entities, direct_speech_groups, tokens):
        pass
