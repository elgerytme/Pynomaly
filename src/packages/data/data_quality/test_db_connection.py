
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = "sqlite:///./data_quality.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

try:
    # Attempt to create tables (if they don't exist) to test connection
    # This requires importing your models, but for a basic connection test, it's not strictly necessary.
    # If you have models defined, you would import them here and call Base.metadata.create_all(bind=engine)
    print("Attempting to connect to the database...")
    with engine.connect() as connection:
        print("Successfully connected to the database.")
except Exception as e:
    print(f"Error connecting to database: {e}")
