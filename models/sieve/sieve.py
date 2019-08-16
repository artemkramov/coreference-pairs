from .direct_speech import DirectSpeech
from .alias import Alias
from abc import ABC, abstractmethod
from models.dictionary import PartOfSpeech
from typing import List
from ..searn.mention import Mention


# Class to apply a set of sieves
class Sieve:
    # Model for working with direct speech
    model_direct_speech = None

    # Model for working with aliases
    model_aliases = None

    # Constructor
    def __init__(self):
        self.model_direct_speech = DirectSpeech()
        self.model_aliases = Alias()

    # Find direct speech groups among tokens
    def find_direct_speech(self, tokens):
        return self.model_direct_speech.find_direct_speech_groups(tokens)

    # Find redirects for named entities
    def find_aliases(self, entities):
        return self.model_aliases.find_wikipedia_links(entities)

    # Apply all handlers
    @staticmethod
    def apply(entities, direct_speech_groups, tokens, aliases):
        discourse_handler = DiscourseHandler()
        full_symbol_handler = FullSymbolHandler()
        partial_symbol_handler = PartialSymbolHandler()
        alias_handler = AliasHandler()
        discourse_handler.set_next(full_symbol_handler).set_next(partial_symbol_handler).set_next(alias_handler)
        return discourse_handler.handle_request(entities, direct_speech_groups, tokens, aliases)


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
    def handle_request(self, entities, direct_speech_groups, tokens, aliases):
        if self._successor is not None:
            return self._successor.handle_request(entities, direct_speech_groups, tokens, aliases)
        return None


# Handler to analyze direct speech
class DiscourseHandler(Handler):

    def handle_request(self, mentions: "List[Mention]", direct_speech_groups, tokens, aliases):
        # Get first tokens from every entity
        token_first_entity = mentions[0].tokens[0]
        token_second_entity = mentions[1].tokens[0]

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
                        and len(mentions[0].tokens) == 1 and len(mentions[1].tokens) == 1 \
                        and (
                        token_first_entity.Lemmatized.lower() == 'я' or token_first_entity.Lemmatized.lower() == 'ти'):
                    return True

                """ Second rule
                автор і займенник «я» в цитатах автора посилаються на один об'єкт
                автор і всі об’єкти у його висловлюваннях, крім займенника «я», відсіюються
                """
                if token_first_group['role'] != token_second_group['role']:
                    author = mentions[0]
                    pronoun = mentions[1]
                    if token_second_group['role'] == 'author':
                        author = mentions[1]
                        pronoun = mentions[0]
                    if self.is_entity_is_single_author(author, direct_speech_groups, tokens) and len(pronoun.tokens) == 1 \
                            and pronoun.tokens[0].PartOfSpeech == PartOfSpeech.PRONOUN:
                        if pronoun.tokens[0].Lemmatized.lower() == 'я':
                            return True
                        else:
                            return False

                """ Third rule
                різні особові займенники, використані в цитатах одного автора, відсіюються
                """
                if token_first_group['role'] == token_second_group['role'] and token_first_group['role'] == 'direct':
                    if token_first_entity.PartOfSpeech == PartOfSpeech.PRONOUN \
                            and token_second_entity.PartOfSpeech == PartOfSpeech.PRONOUN \
                            and len(mentions[0].tokens) == 1 and len(mentions[1].tokens) == 1 \
                            and token_first_entity.Lemmatized.lower() != token_second_entity.Lemmatized.lower():
                        return False

                """ Fourth rule
                іменникові фрази і займенники «я», «ти», «ми» в одному висловлюванні відсіюються
                """
                if token_first_group['role'] == token_second_group['role'] and token_first_group['role'] == 'direct':
                    if token_first_entity.PartOfSpeech == PartOfSpeech.PRONOUN \
                            or token_second_entity.PartOfSpeech == PartOfSpeech.PRONOUN:
                        pronoun = mentions[0]
                        np = mentions[1]
                        if token_second_entity.PartOfSpeech == PartOfSpeech.PRONOUN:
                            pronoun = mentions[1]
                            np = mentions[0]
                        if len(pronoun.tokens) == 1 and pronoun.tokens[0].Lemmatized.lower() in ["я", "ти", "ми"] \
                                and (np.tokens[0].PartOfSpeech != PartOfSpeech.PRONOUN or len(np.tokens) > 1):
                            return False

        return super().handle_request(mentions, direct_speech_groups, tokens, aliases)

    # Check if the entity is a single entity inside the author part
    @staticmethod
    def is_entity_is_single_author(mention: Mention, direct_speech_groups, tokens):
        # Get direct speech group
        entity_group = direct_speech_groups[mention.tokens[0].WordOrder]
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
        if len(mention.tokens) == counter:
            return True
        return False


# Class to check if entities are equal
class FullSymbolHandler(Handler):

    def handle_request(self, mentions: List[Mention], direct_speech_groups, tokens, aliases):
        # Check if count of tokens is the same
        if len(mentions[0].tokens) == len(mentions[1].tokens):
            is_equal = True

            # Loop through tokens of the first entity
            # and compare them with corresponding tokens of the second entity
            # Check if both tokens are proper names
            for idx, tokens_first_entity in enumerate(mentions[0].tokens):
                if not (tokens_first_entity.IsProperName and mentions[1].tokens[idx].IsProperName
                        and tokens_first_entity.Lemmatized.lower() == mentions[1].tokens[idx].Lemmatized.lower()):
                    is_equal = False
                    break
            if is_equal:
                return True
        return super().handle_request(mentions, direct_speech_groups, tokens, aliases)


# Class to handle entities with partial conformity
class PartialSymbolHandler(Handler):

    def handle_request(self, mentions: List[Mention], direct_speech_groups, tokens, aliases):

        # Find head word and entity string of the first and second entities
        head_word_first, words_first = self.find_head_word_with_preceding(mentions[0].tokens)
        head_word_second, words_second = self.find_head_word_with_preceding(mentions[1].tokens)

        # If both head words are proper names and preceding parts are equal
        # than set this pair as a co-referent
        if head_word_first.IsProperName and head_word_second.IsProperName and words_first == words_second:
            return True
        return super().handle_request(mentions, direct_speech_groups, tokens, aliases)

    # Find head word and get string of all preceding words of the head word
    @staticmethod
    def find_head_word_with_preceding(tokens):
        words = []
        head_word = tokens[-1]

        # Loop through tokens of the entity
        for token in tokens:

            # Append lemmatized version of each token
            words.append(token.Lemmatized.lower())

            # If we find a head word than assign it and stop loop
            if token.IsHeadWord:
                head_word = token
                break

        # Form string of entity's words
        entity_string = ' '.join(words)
        return head_word, entity_string


# Class to check if both entities link to the same Wikipedia article
class AliasHandler(Handler):

    def handle_request(self, mentions: List[Mention], direct_speech_groups, tokens, aliases):
        first_entity_id = mentions[0].tokens[0].EntityID
        second_entity_id = mentions[1].tokens[0].EntityID

        # Check if both entities have links in Wikipedia
        if first_entity_id in aliases and second_entity_id in aliases:
            if self.is_entity_link_to_another(aliases[first_entity_id],
                                              aliases[second_entity_id]) \
                    or self.is_entity_link_to_another(aliases[second_entity_id], aliases[first_entity_id]):
                return True
        return super().handle_request(mentions, direct_speech_groups, tokens, aliases)

    # Check if one entity is link to another
    @staticmethod
    def is_entity_link_to_another(first_alias, second_alias):
        if "redirects" in first_alias:
            for redirect in first_alias["redirects"]:
                if redirect["pageid"] == second_alias["pageid"]:
                    return True
        return False
