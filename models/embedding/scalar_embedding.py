from models.embedding.base_embedding import BaseEmbedding
from models.dictionary import PartOfSpeech
from sklearn.feature_extraction.text import TfidfVectorizer


class ScalarEmbedding(BaseEmbedding):

    # TF-IDF dictionary
    tfidf = {}

    # Reset dictionary of TF-IDF measures
    def reset_tfidf(self):
        self.tfidf = {}

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
            else:
                sentences[idx].append(token.Lemmatized.lower())

        self.reset_tfidf()
        tfidf = TfidfVectorizer(
            analyzer='word',
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
        first_item_head = self.get_head_word_index(first_item)
        second_item_head = self.get_head_word_index(second_item)

        vector.append(int(self.check_if_pronoun(first_item)))


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
