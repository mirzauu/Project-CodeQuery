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
    email_verified = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    last_login_at = Column(TIMESTAMP(timezone=True), default=func.now())



class Project(Base):
    __tablename__ = "projects"

    id = Column(Text, primary_key=True)
    properties = Column(BYTEA)
    repo_name = Column(Text)
    repo_path = Column(Text, nullable=True)
    branch_name = Column(Text)
    user_id = Column(
        String(255), ForeignKey("users.uid", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    commit_id = Column(String(255))
    is_deleted = Column(Boolean, default=False)
    updated_at = Column(
        TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now()
    )
    status = Column(String(255), default="created")

    __table_args__ = (
        ForeignKeyConstraint(["user_id"], ["users.uid"], ondelete="CASCADE"),
        CheckConstraint(
            "status IN ('submitted', 'cloned', 'parsed', 'ready', 'error')",
            name="check_status",
        ),
    )

    # Project relationships
    user = relationship("User", back_populates="projects")
    search_indices = relationship("SearchIndex", back_populates="project")
    tasks = relationship("Task", back_populates="project")

    @hybrid_property
    def conversations(self):
        from modules.db.postgress_db import SessionLocal
        from app.modules.conversations.conversation.conversation_model import (
            Conversation,
        )

        with SessionLocal() as session:
            return (
                session.query(Conversation)
                .filter(Conversation.project_ids.any(self.id))
                .all()
            )

