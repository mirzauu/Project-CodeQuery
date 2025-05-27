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
