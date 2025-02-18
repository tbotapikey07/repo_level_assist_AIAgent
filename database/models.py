from sqlalchemy import Column, Integer, String, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class AgentAction(Base):
    """Model to track agent actions and their results."""
    
    __tablename__ = 'agent_actions'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String, nullable=False)
    action_type = Column(String, nullable=False)
    input_data = Column(JSON)
    output_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default='completed')
    error = Column(String, nullable=True)

def init_db(db_path: str = 'sqlite:///codezczar.db'):
    """Initialize the database and create tables.
    
    Args:
        db_path: Path to the SQLite database
    """
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
