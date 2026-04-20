import uuid
from sqlalchemy import Column, Boolean, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from database import Base

class Subject(Base):
    __tablename__ = "subjects"

    # Для SQLite используем String, так как он не знает типа UUID напрямую
    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    pin_hash       = Column(String, nullable=False)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
    consent_given  = Column(Boolean, default=True)
    study_group    = Column(String, nullable=True)