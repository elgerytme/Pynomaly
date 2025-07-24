import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Construct an absolute path to the database file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'data_quality.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()