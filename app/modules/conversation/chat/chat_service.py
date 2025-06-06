import uuid
from datetime import datetime, timezone
from typing import List, Optional,Dict
import logging

from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from .chat_model import (
    Conversation,
    ConversationStatus
)

from .message_schema import (
    MessageCreate,
    MessageInDB,
    MessageStatus,
    MessageType,
    ChatMessageResponse
)
from app.modules.agents.agent_schema import ChatContext
from app.modules.agents.agent_service import AgentsService
from app.modules.provider.provider_service import ProviderService


logger = logging.getLogger(__name__)


class ConversationServiceError(Exception):
    pass


class ConversationNotFoundError(ConversationServiceError):
    pass


class MessageNotFoundError(ConversationServiceError):
    pass


class AccessTypeNotFoundError(ConversationServiceError):
    pass


class AccessTypeReadError(ConversationServiceError):
    pass


class ChatService:
    def __init__(self, project_id: str, user_id: str, sql_db: Session, mongo_db: AsyncIOMotorDatabase):
        self.project_id = project_id
        self.user_id = user_id
        self.sql_db = sql_db
        self.agent_service = AgentsService(sql_db,user_id,ProviderService(sql_db))  # Placeholder for agent service, to be set externally
        self.mongo_collection = mongo_db["chat"]
        self.conversation: Optional[Conversation] = None
        self.message_buffer: Dict[str, Dict[str, str]] = {}

    def init(self):
        self.conversation = self._get_or_create_conversation()

    def _get_or_create_conversation(self) -> Conversation:
        query = (
            select(Conversation)
            .filter(
                Conversation.user_id == self.user_id,
                Conversation.project_ids.any(self.project_id),
                Conversation.status == ConversationStatus.ACTIVE,
            )
        )
        result = self.sql_db.execute(query)
        conversation = result.scalar_one_or_none()

        if conversation:
            return conversation

        new_conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=self.user_id,
            project_ids=[self.project_id],
        )
        self.sql_db.add(new_conversation)
        self.sql_db.commit()
        self.sql_db.refresh(new_conversation)
        return new_conversation

    async def store_message(self, message: MessageCreate) -> str:
        doc = {
            "conversation_id": message.conversation_id,
            "content": message.content,
            "type": message.type,
            "sender_id": message.sender_id,
            "citations": message.citations,
            "status": "ACTIVE",
            "created_at": datetime.utcnow()
        }
        result = await self.mongo_collection.insert_one(doc)
        return str(result.inserted_id)
    
    async def post_message(self, message: MessageCreate, project_id: str, user_id: str):
        """
        Handles the posting of a user message:
        1. Buffers and stores the message using chunking.
        2. Triggers AI response generation and streams the result.
        3. Yields the final AI response as a ChatMessageResponse.
        """
        try:
            # Step 1: Add message to buffer
            self.add_message_chunk(
                conversation_id=message.conversation_id,
                content=message.content,
                message_type=message.type,
                sender_id=message.sender_id,
                citations=message.citations,
            )

            # Step 2: Flush message to database
            await self.flush_message_buffer(
                conversation_id=message.conversation_id,
                message_type=message.type,
                sender_id=message.sender_id,
            )

            logger.info(f"Stored message for conversation: {message.conversation_id}")


            async for chunk in self._generate_and_stream_ai_response(
                message.content,
                message.conversation_id,
                user_id,
                project_id=project_id,
            ):
                yield chunk
               

          

        except Exception as e:
            logger.error(f"Failed to post message for conversation {message.conversation_id}: {e}", exc_info=True)
            raise


    async def delete_message(self, message_id: str) -> bool:
        result = await self.mongo_collection.update_one(
            {"_id": message_id, "conversation_id": self.conversation.id},
            {"$set": {"status": MessageStatus.DELETED}}
        )
        return result.modified_count > 0

    async def get_chat_history(self, include_deleted: bool = False) -> List[MessageInDB]:
        query = {"conversation_id": self.conversation.id}
        if not include_deleted:
            query["status"] = MessageStatus.ACTIVE

        cursor = self.mongo_collection.find(query).sort("created_at", 1)
        messages = []
        async for doc in cursor:
            messages.append(MessageInDB(**doc))
        return messages
    
    async def _generate_and_stream_ai_response(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        project_id: str,
    ):
        print("project",project_id)
        """
        Streams AI-generated responses using agent_service, 
        buffering each chunk and flushing at the end.
        """
        try:
            # Step 1: Get and validate history
            try:
                history = await self.get_session_history(user_id)
                validated_history = [
                    f"{msg.type}: {msg.content}" if hasattr(msg, "content") else str(msg)
                    for msg in history
                ]
            except Exception as e:
                logger.error(f"Failed to retrieve chat history: {e}", exc_info=True)
                raise ConversationServiceError("Failed to get chat history")

            # Step 2: Start streaming from the agent service
            logger.debug(f"Calling execute_stream with query: {query} and history: {validated_history[-8:]}")
            res = self.agent_service.execute_stream(
                ChatContext(
                    project_id=str(project_id),
                    history=validated_history[-8:],  # Only the last 8 messages
                    query=query,  
                )
            )

            async for chunk in res:
                print(f"Received chunk: {chunk}")
                content = getattr(chunk, "response", "")
                citations = getattr(chunk, "citations", [])
                self.add_message_chunk(
                    conversation_id=conversation_id,
                    content=content,
                    message_type=MessageType.AI_GENERATED,
                    citations=citations,
                )

                yield ChatMessageResponse(
                    message=chunk.response,
                    citations=chunk.citations,
                    tool_calls=[
                        tool_call.model_dump_json()
                        for tool_call in getattr(chunk, "tool_calls", [])
                    ],
                )
                print(f"Streaming chunk: {chunk}")

            # Step 3: Flush all collected AI-generated chunks
            await self.flush_message_buffer(
                conversation_id=conversation_id,
                message_type=MessageType.AI_GENERATED,
            )

            logger.info(
                f"Generated and streamed AI response for conversation {conversation_id} and user {user_id}"
            )

        except Exception as e:
            logger.error(
                f"Failed to generate and stream AI response for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise ConversationServiceError(
                "Failed to generate and stream AI response."
            ) from e



    
    def add_message_chunk(
        self,
        conversation_id: str,
        content: str,
        message_type: MessageType,
        sender_id: Optional[str] = None,
        citations: Optional[List[str]] = None,
    ):
        if conversation_id not in self.message_buffer:
            self.message_buffer[conversation_id] = {"content": "", "citations": []}
        self.message_buffer[conversation_id]["content"] += content
        if citations:
            self.message_buffer[conversation_id]["citations"].extend(citations)
        logger.debug(f"Added message chunk to buffer for conversation: {conversation_id}")

    async def flush_message_buffer(
        self,
        conversation_id: str,
        message_type: MessageType,
        sender_id: Optional[str] = None,
    ):
        if (
            conversation_id in self.message_buffer
            and self.message_buffer[conversation_id]["content"]
        ):
            content = self.message_buffer[conversation_id]["content"]
            citations = self.message_buffer[conversation_id]["citations"]

            message_doc = {
                "conversation_id": conversation_id,
                "content": content,
                "type": message_type,
                "sender_id": sender_id if message_type == MessageType.HUMAN else None,
                "citations": list(set(citations)) if citations else [],
                "status": "ACTIVE",
                "created_at": datetime.now(timezone.utc),
            }

            await self.mongo_collection.insert_one(message_doc)
            self.message_buffer[conversation_id] = {"content": "", "citations": []}
            logger.info(f"Flushed message buffer for conversation: {conversation_id}")

    async def get_session_history(self, user_id: str) -> List[BaseMessage]:
        try:
            cursor = self.mongo_collection.find(
                {
                    "conversation_id": self.conversation.id,
                    "status": "ACTIVE"
                }
            ).sort("created_at", 1)  # Sort by oldest first

            history = []
            async for doc in cursor:
                if doc["type"] == MessageType.HUMAN:
                    history.append(HumanMessage(content=doc["content"]))
                else:
                    history.append(AIMessage(content=doc["content"]))

            logger.info(f"Retrieved session history for conversation: ")
            return history

        except Exception as e:
            logger.error(f"Failed to retrieve session history: {e}")
            return []        