import enum

from sqlalchemy import ARRAY, TIMESTAMP, Column
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy import ForeignKey, String, func
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from app.core.db.postgress_db import Base



class ConversationStatus(enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Visibility(enum.Enum):
    PRIVATE = "private"
    PUBLIC = "public"


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), ForeignKey("users.uid", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    status = Column(SQLAEnum(ConversationStatus), default=ConversationStatus.ACTIVE, nullable=False)
    project_ids = Column(ARRAY(String), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    shared_with_emails = Column(ARRAY(String), nullable=True)
    visibility = Column(SQLAEnum(Visibility), default=Visibility.PRIVATE, nullable=True)

    user = relationship("User", back_populates="conversations")
