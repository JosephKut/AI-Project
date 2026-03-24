from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from src.config import DB_URI

Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    id = Column(Integer, primary_key=True)
    district = Column(String(100))
    date = Column(DateTime, default=datetime.utcnow)
    features = Column(JSON)
    prediction = Column(String(50))
    confidence = Column(Float)

class TrainingLog(Base):
    __tablename__ = 'training_log'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_type = Column(String(50))
    metrics = Column(JSON)

engine = create_engine(DB_URI, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == '__main__':
    init_db()
    print('Database initialized: %s' % DB_URI)
