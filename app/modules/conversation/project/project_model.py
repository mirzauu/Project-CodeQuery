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
    Integer
)
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from app.core.db.postgress_db import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Text, primary_key=True)
    properties = Column(String, nullable=True)  # or BYTEA if you're storing binary
    repo_name = Column(Text)
    repo_path = Column(Text, nullable=True)
    repo_url = Column(Text, nullable=True)
    user_id = Column(String(255), ForeignKey("users.uid", ondelete="CASCADE"), nullable=False)
    is_deleted = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now())
    status = Column(String(255), default="created")

    user = relationship("User", back_populates="projects")
    search_indices = relationship("SearchIndex", back_populates="project", cascade="all, delete-orphan")


    __table_args__ = (
        ForeignKeyConstraint(["user_id"], ["users.uid"], ondelete="CASCADE"),
        CheckConstraint("status IN ('submitted', 'cloned', 'parsed', 'ready', 'error')", name="check_status"),
    )



class SearchIndex(Base):
    __tablename__ = "search_indices"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Text, ForeignKey("projects.id"), index=True)
    node_id = Column(String, index=True)
    name = Column(String, index=True)
    file_path = Column(String, index=True)
    content = Column(Text)

    project = relationship("Project", back_populates="search_indices")
