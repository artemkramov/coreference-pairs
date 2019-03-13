from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Connect to the database and create session marker for making of the queries
engine = create_engine('mysql://root:123456@192.168.0.123/coreference_dump?charset=utf8')
session = sessionmaker()
session.configure(bind=engine)
declarative_base().metadata.create_all(engine)
