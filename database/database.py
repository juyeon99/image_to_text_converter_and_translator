from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DB_URL = "sqlite:///imagetotext.sqlite3"

engine = create_engine(DB_URL, connect_args={"check_same_thread":False})

session_local = sessionmaker(autoflush=False, autocommit=False, bind=engine)

Base = declarative_base()