import re
import uuid
from ..db.word import DbWord
from .universal_dependency_model import UniversalDependencyModel
import csv
import dateparser.search as DateSearch
from os.path import join
import os
from mitie import *


class NLP:
    # Encoding of the files
    ENCODING = 'utf-8'

    # Gazetteer data
    gazetteer = []

    # Universal dependency model
    ud_model = None

    # NER model
    ner = None

    def __init__(self):

        directory = os.path.dirname(os.path.realpath(__file__))

        # Load NER model
        print('Loading NER model...')
        self.ner = named_entity_extractor(join(directory, 'uk_model.dat'))

        # Load gazetteers
        print('Loading gazetteer...')

        with open(join(directory, 'gazetteers\\gazetteer.csv'), encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.gazetteer.append({
                    'type': 'gazetteer',
                    'text': row[0]
                })

        # Load UD parser model
        self.ud_model = UniversalDependencyModel(join(directory, 'ukrainian-iu-ud-2.3-181115.udpipe'))

    # Apply gazetteer for text and try ro find named entities from it
    def preprocess_text_with_gazetteer(self, text):
        # Array with terms which were found in the text
        gazetteer_entity_pointers = []

        # List with ranges of all found gazetteer entities
        gazetteer_entities = []

        # Try to detect dates inside the raw text
        # 'uk' parameter means that we're working with Ukrainian text
        date_matches = None
        try:
            date_matches = DateSearch.search_dates(text, languages=['uk'])
        except Exception:
            date_matches = None
        if date_matches is None:
            dates = []
        else:
            dates = [{
                'type': 'date',
                'text': date[0]
            } for date in date_matches]

        # Copy gazetteer value and append detected dates
        term_db = self.gazetteer.copy()
        term_db.extend(dates)

        # Loop through gazetteer terms and try to find matches
        for term in term_db:
            # Apply regular expression and generate list of tuples (start, end) with all occurrences
            try:
                term_occurrences = [(m.start(), m.start() + len(term['text'])) for m in re.finditer(term['text'], text)]
            except Exception:
                term_occurrences = []
            if len(term_occurrences) > 0:
                # Parse term to form the tokens which will replace UUID
                tokens = []
                term_sentences = self.ud_model.tokenize(term['text'])
                for s in term_sentences:
                    i = 0
                    while i < len(s.words):
                        # Omit <root> tag
                        if s.words[i].form != '<root>':
                            tokens.append(s.words[i].form)
                        i += 1

                # Add entity to gazetteer vocabulary
                is_proper_name = True
                is_merge_allowed = False
                if term['type'] != 'gazetteer':
                    is_proper_name = False
                if not (any(pointer['term'] == term['text'] for pointer in gazetteer_entity_pointers)):
                    gazetteer_entity_pointers.append({
                        'term': term['text'],
                        'tokens': tokens,
                        'is_proper_name': is_proper_name,
                        'is_merge_allowed': is_merge_allowed
                    })
        # Tokenize the whole text
        sentences = self.ud_model.tokenize(text)

        # Loop through each found gazetteer entity
        for gazetteer_entity_pointer in gazetteer_entity_pointers:
            window_size = len(gazetteer_entity_pointer['tokens'])
            sentence_offset = 0
            # Loop though all words of each sentence
            # and check if window is equal to current clique
            for s in sentences:
                i = 0

                # Slide window across all text
                while i < len(s.words) - window_size:
                    # Set start and finish position of current clique
                    j = i
                    is_clique_entity = True
                    clique_end = i + window_size

                    # Loop through clique and compare its tokens with gazetteer entity tokens
                    while j < clique_end:
                        if s.words[j].form != gazetteer_entity_pointer['tokens'][j - i]:
                            is_clique_entity = False
                            break
                        j += 1
                    if is_clique_entity:
                        gazetteer_entities.append({
                            'items': range(i + sentence_offset, clique_end + sentence_offset),
                            'head_word': None,
                            'is_proper_name': gazetteer_entity_pointer['is_proper_name'],
                            'is_merge_allowed': gazetteer_entity_pointer['is_merge_allowed']
                        })
                    i += 1
                sentence_offset += len(s.words)

        return gazetteer_entities

    # Extract all named entities and nouns/pronouns from text
    def extract_entities(self, text):
        tokens = []
        tagged_words = []

        # Apply gazetteer for the text and tokenize text
        gazetteer_entities = self.preprocess_text_with_gazetteer(text)

        # Parse text with UD
        sentences = self.ud_model.tokenize(text)

        # Init word counter to write worder order
        word_counter = 0

        for s in sentences:
            self.ud_model.tag(s)
            self.ud_model.parse(s)

            i = 0
            while i < len(s.words):
                word = s.words[i]
                raw_word = word.form.strip()
                tokens.append(raw_word)
                word_tag = word.feats
                if i == len(s.words) - 1 and raw_word == '.':
                    word_tag = './SENT_END'

                is_plural, gender = self.ud_model.parse_tag(word_tag)

                tagged_word = DbWord()
                tagged_word.RawText = raw_word
                tagged_word.WordOrder = word_counter
                tagged_word.PartOfSpeech = word.upostag
                tagged_word.Lemmatized = word.lemma
                tagged_word.IsPlural = is_plural
                tagged_word.IsProperName = False
                tagged_word.IsHeadWord = False
                tagged_word.Gender = gender
                tagged_word.RawTagString = word_tag

                # tagged_word = {
                #     'word': raw_word,
                #     'lemma': word.lemma,
                #     'tag': word_tag,
                #     'isEntity': False,
                #     'isProperName': False,
                #     'isHeadWord': False,
                #     'groupID': None,
                #     'groupLength': None,
                #     'groupWord': None,
                #     'pos': word.upostag
                # }
                tagged_words.append(tagged_word)
                i += 1
                word_counter += 1

        # Apply NER model for searching of named entities
        named_entities_models = self.ner.extract_entities(tokens)

        # Add named entities that were found
        # But check if their does'nt intersect with gazetteer data
        named_entities = gazetteer_entities[:]
        for named_entities_model in named_entities_models:
            is_entity_new = True
            for gazetteer_entity in gazetteer_entities:
                if len(set(gazetteer_entity['items']).intersection(named_entities_model[0])) > 0:
                    is_entity_new = False
                    break
            if is_entity_new and named_entities_model[2] > 0.6:

                # Check if created named entity doesn't share two sentences
                # Find out if named entity contains the symbol of sentence separation
                index_to_divide = -1
                parts = [named_entities_model[0]]
                for idx in named_entities_model[0]:
                    if tagged_words[idx].RawTagString == './SENT_END':
                        index_to_divide = idx
                        break

                # If the sentence division was found
                # Than split range into 2 parts around that symbol
                if index_to_divide > -1:

                    # Left part before symbol
                    left_part = range(named_entities_model[0][0], index_to_divide)

                    # Declare start and finish indexes for right part
                    right_part_start = index_to_divide + 1
                    if tagged_words[index_to_divide + 1].RawText == '<root>':
                        right_part_start += 1
                    right_part_end = named_entities_model[0][-1] + 1
                    if right_part_start < right_part_end:
                        right_part = range(right_part_start, right_part_end)
                    else:
                        right_part = []

                    parts = [left_part, right_part]
                for part in parts:
                    if isinstance(part, (range,)):

                        if len(part) > 0:
                            # Check if start and finish items of group doesn't contain <root>
                            start_number = part[0]
                            finish_number = part[-1] + 1
                            if tagged_words[start_number].RawText == '<root>':
                                start_number += 1
                            if finish_number < len(tagged_words) and tagged_words[finish_number].RawText == '<root>':
                                finish_number -= 1

                            if start_number < finish_number:
                                named_entities.append({
                                    'items': range(start_number, finish_number),
                                    'head_word': None,
                                    'is_proper_name': True,
                                    'is_merge_allowed': False
                                })

        # Extract noun phrases from text but with excluding of the named entities
        ud_groups, ud_levels = self.ud_model.extract_noun_phrases(sentences, [])
        for ud_group in ud_groups:
            named_entities.append({
                'items': ud_group['items'],
                'head_word': ud_group['head_word'],
                'is_proper_name': ud_group['is_proper_name']
            })

        # Sort named entities by the start range value
        named_entities.sort(key=lambda group: group['items'][0])

        # Merge all named entities that has intersection
        current_entity_idx = 0
        named_entities_aligned = []
        exclude_idx = []
        while current_entity_idx < len(named_entities):
            i = current_entity_idx
            current_group = named_entities[current_entity_idx]

            # Set head word if it's a None
            # Head word we set as the highest element corresponding to tree
            if current_group['head_word'] is None and len(current_group['items']) > 0:
                current_group_head_word = current_group['items'][0]
                for word_id in current_group['items']:
                    if ud_levels[word_id] < ud_levels[current_group_head_word]:
                        current_group_head_word = word_id
                current_group['head_word'] = current_group_head_word

            if len(current_group['items']) > 0 and (not (current_entity_idx in exclude_idx)):
                while i < len(named_entities) - 1:

                    is_merge_allowed = True
                    if (not current_group.get('is_merge_allowed', True)) or (
                            not named_entities[i + 1].get('is_merge_allowed', True)):
                        is_merge_allowed = False

                    # Check for intersection
                    if len(set(current_group['items']).intersection(named_entities[i + 1]['items'])) > 0:

                        if not is_merge_allowed:
                            current_group_set = set(current_group['items'])
                            next_named_entity_set = set(named_entities[i + 1]['items'])
                            if not current_group.get('is_merge_allowed', True):
                                next_named_entity_set.difference_update(current_group_set)
                            else:
                                current_group_set.difference_update(next_named_entity_set)
                            if len(current_group_set) == 0:
                                current_group['items'] = []
                                break
                            if len(next_named_entity_set) == 0:
                                named_entities[i + 1]['items'] = []
                            tmp_items = self.fix_divided_set(sorted(current_group_set))
                            if len(tmp_items) > 0:
                                current_group['items'] = range(tmp_items[0], tmp_items[-1] + 1)
                            if len(next_named_entity_set) > 0:
                                tmp_items = self.fix_divided_set(sorted(next_named_entity_set))
                                if len(tmp_items) > 0:
                                    named_entities[i + 1]['items'] = range(tmp_items[0], tmp_items[-1] + 1)
                        else:
                            # Convert range to sets
                            # And perform union operation on intersected ranges
                            items = list(set(current_group['items']).union(set(named_entities[i + 1]['items'])))
                            items.sort()
                            group_items = range(items[0], items[-1] + 1)
                            current_group['items'] = group_items
                            exclude_idx.append(i + 1)

                            # Compare head words of groups
                            # If the head word of the next group is determined
                            # and it's located above the head word of the current group
                            # than we reset head word and proper name flag for current group
                            if not (named_entities[i + 1]['head_word'] is None):
                                if current_group['head_word'] is None or ud_levels[
                                    named_entities[i + 1]['head_word']] < \
                                        current_group['head_word']:
                                    current_group['head_word'] = named_entities[i + 1]['head_word']
                                    current_group['is_proper_name'] = named_entities[i + 1]['is_proper_name']
                        i += 1
                    else:
                        break
                if len(current_group['items']) > 0:
                    named_entities_aligned.append(current_group)
                if current_entity_idx == len(named_entities) - 1:
                    break
            current_entity_idx += 1

        named_entities = named_entities_aligned

        # Form list of positions which are used in the named entity
        # It is used for ignoring of them further
        exclude_entities_index = []
        named_entities_range = []
        for idx, named_entity in enumerate(named_entities):

            # Set common group ID for all words of the named entity
            group_id = idx
            named_entities_index = []

            for i in named_entity['items']:

                # Append index of the each token of named entity
                # Also append it to exclude list for the following processing
                named_entities_index.append(i)
                exclude_entities_index.append(i)

                # Set it as entity, add common group and set proper name attribute
                tagged_words[i].EntityID = group_id
                if named_entity['is_proper_name']:
                    tagged_words[i].IsProperName = True

                # Set head word flag
                if i == named_entity['head_word'] and (not (named_entity['head_word'] is None)):
                    tagged_words[i].IsHeadWord = True

            # Convert range to list for JSON serialization
            named_entities_range.append(named_entities_index)
        entities = []
        for position, token in enumerate(tagged_words):
            # Check if word isn't a part of named entity
            if not (position in exclude_entities_index):

                # Tag string for the detection of the part of speech and its attributes
                tag_string = token.RawTagString

                # Check if the entity is personal pronoun
                is_personal_pronoun = False
                if token.PartOfSpeech == 'PRON' and tag_string.find("PronType=Prs") > -1:
                    is_personal_pronoun = True

                if is_personal_pronoun or token.PartOfSpeech == 'PROPN' or token.PartOfSpeech == 'X':
                    entities.append(position)
                    tagged_words[position].EntityID = uuid.uuid4()
                    tagged_words[position].IsHeadWord = True

                # Check if proper name was detected
                if token.PartOfSpeech == 'PROPN':
                    tagged_words[position].IsHeadWord = True

        summary = {
            "tokens": tagged_words
        }
        return summary

    @staticmethod
    def fix_divided_set(group_list):
        group_aligned = set()
        i = 0
        if len(group_list) == 1:
            return group_list
        while i < len(group_list) - 1:
            if group_list[i] + 1 == group_list[i + 1]:
                group_aligned.update([group_list[i], group_list[i + 1]])
            else:
                break
            i += 1
        return sorted(group_aligned)

    # Save token with the given parameters
    def save_token(self, parameters):
        # Create new word entity
        token = DbWord()

        # Set all dictionary attributes to new word object
        attributes = list(parameters.keys())
        for attribute in attributes:
            setattr(token, attribute, parameters[attribute])

        is_plural, gender = self.ud_model.parse_tag(parameters['RawTagString'])
        token.IsPlural = is_plural
        token.Gender = gender

    # Save all tokens with corresponding groups
    def save_tokens(self, entities, remote_ip_address):
        # Word order in the document
        word_order = 0

        # Dictionary of cluster groups
        cluster_groups = {}

        # Generate unique document ID
        document_id = uuid.uuid4()

        # Loop through collection of entities
        for entity in entities:
            entity_id = None
            cluster_id = None

            if entity['pos'] == '<root>':
                continue

            # Check if entity is located inside some cluster
            if not (entity['clusterID'] is None):
                # Create new cluster group inside dictionary if it doesn't exist
                if not (entity['clusterID'] in cluster_groups):
                    cluster_groups[entity['clusterID']] = uuid.uuid4()
                cluster_id = cluster_groups[entity['clusterID']]

            # Check if entity contains few words
            if len(entity['groupWords']) > 0:

                # Generate common entity ID for all words of the entity
                # and save each inner word
                entity_id = uuid.uuid4()
                for token in entity['groupWords']:
                    parameters = {
                        'RawText': token['word'],
                        'DocumentID': document_id,
                        'WordOrder': word_order,
                        'Lemmatized': token['lemma'],
                        'EntityID': entity_id,
                        'CoreferenceGroupID': cluster_id,
                        'RawTagString': token['tag'],
                        'IsHeadWord': token['isHeadWord'],
                        'IsProperName': token['isProperName'],
                        'PartOfSpeech': token['pos'],
                        'RemoteIPAddress': remote_ip_address
                    }
                    self.save_token(parameters)
                    word_order += 1
            else:
                # Generate unique ID if the input token is entity (noun or pronoun)
                # and save it
                if entity['isEntity']:
                    entity_id = uuid.uuid4()
                parameters = {
                    'RawText': entity['word'],
                    'DocumentID': document_id,
                    'WordOrder': word_order,
                    'Lemmatized': entity['lemma'],
                    'EntityID': entity_id,
                    'CoreferenceGroupID': cluster_id,
                    'RawTagString': entity['tag'],
                    'IsHeadWord': entity['isHeadWord'],
                    'IsProperName': entity['isProperName'],
                    'PartOfSpeech': entity['pos'],
                    'RemoteIPAddress': remote_ip_address
                }
                self.save_token(parameters)
                word_order += 1
