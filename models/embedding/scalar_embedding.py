from models.embedding.base_embedding import BaseEmbedding
from models.dictionary import PartOfSpeech
from sklearn.feature_extraction.text import TfidfVectorizer


class ScalarEmbedding(BaseEmbedding):

    # TF-IDF dictionary
    tfidf = {}

    # Tokens of the current document
    tokens = []

    # Reset dictionary of TF-IDF measures
    def reset_tfidf(self):
        self.tfidf = {}
        self.tokens = []

    # Function to perform tokenization of TF-IDF
    def dummy_fun(self, tokens):
        return tokens

    # Evaluate TF-IDF measure for all tokens
    def evaluate_tfidf(self, tokens):

        # Firstly split tokens into sentences to perform TF-IDF algorithm
        sentences = [[]]
        idx = 0
        is_prev_sentence_end = False
        for token in tokens:
            if is_prev_sentence_end:
                sentences.append([])
                idx += 1
                is_prev_sentence_end = False
            if token.RawTagString == './SENT_END':
                is_prev_sentence_end = True
            sentences[idx].append(token.Lemmatized.lower())

        self.reset_tfidf()
        self.tokens = tokens
        tfidf = TfidfVectorizer(
            analyzer='word',
            use_idf=True,
            tokenizer=self.dummy_fun,
            preprocessor=self.dummy_fun)
        tfidf.fit(sentences)
        self.tfidf = tfidf.vocabulary_

    def pair2vec(self, pair_words):
        vector = []

        # Split array into two items
        first_item = pair_words[0]
        second_item = pair_words[1]

        # Detect head words of each item
        first_item_head = first_item[self.get_head_word_index(first_item)]
        second_item_head = second_item[self.get_head_word_index(second_item)]

        # Append coordinates with features such as a pronoun detection
        # and TF-IDF value
        vector.append(int(self.check_if_pronoun(first_item_head)))
        vector.append(self.tfidf[first_item_head.Lemmatized.lower()])
        vector.append(int(self.check_if_pronoun(second_item_head)))
        vector.append(self.tfidf[second_item_head.Lemmatized.lower()])

        # Check if the second item is like 'той', 'та' etc.
        # It means that such pronoun has type 'Dem' (Demonstrative)
        is_second_item_dem = 0
        if self.check_if_pronoun(second_item_head) and self.fetch_morphological_feature(second_item_head.RawTagString,
                                                                                        'PronType')[0] == 'Dem':
            is_second_item_dem = 1
        vector.append(is_second_item_dem)

        # Get count of words between entities
        if first_item[0].WordOrder < second_item[0].WordOrder:
            start_index = first_item[0].WordOrder
            last_index = second_item[-1].WordOrder
        else:
            start_index = second_item[0].WordOrder
            last_index = first_item[-1].WordOrder
        vector.append(abs(last_index - start_index))

        # Get count of entities between entities
        entity_count = 0
        word_interval = self.tokens[start_index:last_index]
        for token in word_interval:
            if not (token.EntityID is None):
                entity_count += 1
        vector.append(entity_count)

        # Check if lemmatized versions of strings equal
        is_equal = True
        if len(first_item) == len(second_item):
            for idx, item in enumerate(first_item):
                if item.Lemmatized != second_item[idx].Lemmatized:
                    is_equal = False
                    break
        vector.append(int(is_equal))

        # Check if number of entities is the same
        vector.append(int(second_item_head.IsPlural == first_item_head.IsPlural))

        # Check if the gender is the same
        vector.append(int(second_item_head.Gender == first_item_head.Gender))

        # Check if both entities are proper names
        vector.append(int(first_item_head.IsProperName and second_item_head.IsProperName))



    # Parse tag string (like Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing
    @staticmethod
    def parse_morphological_tag(tag_string):
        # Split by delimiter to separate each string
        morphology_strings = tag_string.split('|')
        morphology_attributes = []
        for morphology_string in morphology_strings:
            # Split each string to fetch attribute and its value
            morphology_attribute = morphology_string.split('=')
            morphology_attributes.append(morphology_attribute)
        return morphology_attributes

    # Fetch morphological feature by the given name
    def fetch_morphological_feature(self, tag_string, feature_name):
        morphology_attributes = self.parse_morphological_tag(tag_string)
        return [attribute_data[1] for attribute_data in morphology_attributes if attribute_data[0] == feature_name]

    # Find head word index of the entity
    @staticmethod
    def get_head_word_index(item):
        index = 0
        for idx, token in enumerate(item):
            if token.IsHeadWord:
                return idx
        return index

    @staticmethod
    def check_if_pronoun(token):
        return token.PartOfSpeech == PartOfSpeech.PRONOUN
