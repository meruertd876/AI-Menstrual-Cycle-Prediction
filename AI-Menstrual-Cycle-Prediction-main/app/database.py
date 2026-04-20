from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import os

# Используем простую базу данных в файле для начала
DATABASE_URL = "sqlite:///./pinkcycle.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()