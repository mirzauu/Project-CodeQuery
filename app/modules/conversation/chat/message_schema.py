from pydantic import BaseModel, Field
from typing import Optional, List, Literal,Any
from datetime import datetime


class MessageType(str):
    HUMAN = "HUMAN"
    AI_GENERATED = "AI_GENERATED"
    SYSTEM_GENERATED = "SYSTEM_GENERATED"

class MessageStatus(str):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"

class MessageCreate(BaseModel):
    conversation_id: str
    content: str
    type: Literal["HUMAN", "AI_GENERATED", "SYSTEM_GENERATED"]
    sender_id: Optional[str] = None
    citations: Optional[List[str]] = None


class MessageInDB(MessageCreate):
    id: str = Field(alias="_id")
    status: Literal["ACTIVE", "DELETED"] = "ACTIVE"
    created_at: datetime


class MessageInput(BaseModel):
    content: str

class ChatMessageResponse(BaseModel):
    message: str
    citations: List[str]
    tool_calls: List[Any]

class MessageOut(BaseModel):
    sender_type: str  # "HUMAN" or "AI"
    content: str
