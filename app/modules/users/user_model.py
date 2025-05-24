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

from .base_model import Base


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


class Project(Base):
    __tablename__ = "projects"

    id = Column(Text, primary_key=True)
    properties = Column(String)  # or BYTEA if you're storing binary
    repo_name = Column(Text)
    repo_path = Column(Text, nullable=True)
    user_id = Column(String(255), ForeignKey("users.uid", ondelete="CASCADE"), nullable=False)
    commit_id = Column(String(255))
    is_deleted = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now())
    status = Column(String(255), default="created")

    user = relationship("User", back_populates="projects")

    __table_args__ = (
        ForeignKeyConstraint(["user_id"], ["users.uid"], ondelete="CASCADE"),
        CheckConstraint("status IN ('submitted', 'cloned', 'parsed', 'ready', 'error')", name="check_status"),
    )
