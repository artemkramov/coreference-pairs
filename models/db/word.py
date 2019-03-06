from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer

# Set base class for creating of the model
Base = declarative_base()


# Class to represent each word in database
class DbWord(Base):
    __tablename__="word"
    ID = Column(Integer, primary_key=True)
    RawText = Column(String)
    DocumentID = Column(String)
    WordOrder = Column(Integer)
    PartOfSpeech = Column(String)
    Lemmatized = Column(String)
    IsPlural = Column(Integer)
    IsProperName = Column(Integer)
    IsHeadWord = Column(Integer)
    Gender = Column(String)
    EntityID = Column(String)
    RawTagString = Column(String)
    CoreferenceGroupID = Column(String)
    RemoteIPAddress = Column(String)
