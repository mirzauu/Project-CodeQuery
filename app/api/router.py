from fastapi import APIRouter, Depends, HTTPException,status
from sqlalchemy.orm import Session
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse
import json

from app.core.db.postgress_db import get_db
from app.core.db.mongo_db import get_mongo_db, AsyncIOMotorDatabase
from app.modules.conversation.chat.message_schema import MessageCreate,MessageInput,MessageOut
from app.modules.users.user_schema import UserCreate, UserLogin
from app.modules.knowledge.schema.parsing_schema import ParseRequest
from app.modules.users.user_service import UserService
from app.modules.conversation.project.project_service import ProjectService
from app.modules.conversation.chat.chat_service import ChatService
from app.modules.knowledge.parsing_service import ParsingService

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/send-otp")
def send_otp(data: UserCreate, db: Session = Depends(get_db)):
    service = UserService(db)
    return service.send_otp(data)

@router.post("/verify-otp")
def verify_otp(data: UserLogin, db: Session = Depends(get_db)): 
    service = UserService(db)
    try:
        return service.verify_otp_and_login(data)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

@router.post("/parse")
async def parse(data: ParseRequest,user=Depends(UserService.check_auth), db: Session = Depends(get_db)): 
    parse = ParsingService(db,user.uid)
  
    return await parse.parse_validator(data.repo_link,user.uid)


@router.get("/projects")
async def get_projects(user_id=Depends(UserService.check_auth), db=Depends(get_db)):
    service = ProjectService(db)
    return await service.list_projects(user_id.uid)


@router.post("/conversations/{project_id}/message/")
async def send_message(
    project_id: str,
    body: MessageInput,
    user_id=Depends(UserService.check_auth),
    db: Session = Depends(get_db),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    if body.content is None or body.content.strip() == "":
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    try:
        # Init chat service
        chat_service = ChatService(project_id=project_id, user_id=user_id.uid, sql_db=db, mongo_db=mongo_db)
        chat_service.init()

        message_data = MessageCreate(
            conversation_id=chat_service.conversation.id,
            content=body.content,
            type="HUMAN",
            sender_id=user_id.uid
        )

        # Create the async generator
        message_stream = chat_service.post_message(message_data, project_id, user_id.uid)

        async def stream_response():
            async for chunk in message_stream:
                yield (json.dumps(chunk.model_dump()) + "\n")

        return StreamingResponse(stream_response(), media_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send message: {str(e)}"
        )


@router.get("/conversations/{project_id}/history", response_model=List[MessageOut])
async def get_conversation_history(
    project_id: str,
    user_id=Depends(UserService.check_auth),
    db: Session = Depends(get_db),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    try:
        # Initialize chat service
        chat_service = ChatService(project_id=project_id, user_id=user_id.uid, sql_db=db, mongo_db=mongo_db)
        chat_service.init()  # sync call

        # Fetch history using your method
        messages = await chat_service.get_session_history(user_id.uid)

        # Convert LangChain messages (HumanMessage/AIMessage) into response schema
        response = []
        for msg in messages:
            response.append({
                "sender_type": "HUMAN" if msg.__class__.__name__ == "HumanMessage" else "AI",
                "content": msg.content
            })

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )