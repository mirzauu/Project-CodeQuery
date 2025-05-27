from sqlalchemy import TIMESTAMP, Boolean, Column, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from app.core.db.postgress_db import Base
from app.modules.conversation.chat.chat_model import Conversation


class User(Base):
    __tablename__ = "users"

    uid = Column(String(255), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    
    otp_code = Column(String(10), nullable=True)  # store last OTP (hashed in production)
    otp_expiry = Column(TIMESTAMP(timezone=True), nullable=True)  # expires after X minutes
    email_verified = Column(Boolean, default=False)
    
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    last_login_at = Column(TIMESTAMP(timezone=True), default=func.now())

    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

