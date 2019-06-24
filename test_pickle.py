import pickle


class DbWordWeb:

    ID = 0
    RawText = ""
    DocumentID = ""
    WordOrder = ""
    PartOfSpeech = ""
    Lemmatized = ""
    IsPlural = ""
    IsProperName = ""
    IsHeadWord = ""
    Gender = ""
    EntityID = ""
    RawTagString = ""
    CoreferenceGroupID = ""
    RemoteIPAddress = ""


class MentionWeb:

    # Tokens of the entity
    tokens = []

    # ID of cluster
    cluster_id: str = ""

    # Check if it is an entity
    is_entity: bool = False

    def __init__(self, _tokens):
        self.tokens = _tokens.copy()


filename = "dataset_3/data-web-0-0.pkl"
handle = open(filename, 'rb')
items = pickle.load(handle)
handle.close()
