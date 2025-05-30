from fastapi import APIRouter, Depends, HTTPException,status
from sqlalchemy.orm import Session
from app.core.db.postgress_db import get_db

from app.core.db.mongo_db import get_mongo_db, AsyncIOMotorDatabase
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.conversation.chat.message_schema import MessageCreate,MessageInput
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


@router.get("/me")
def get_current_user(user=Depends(UserService.check_auth)):
    return {
        "uid": user.uid,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
    }


@router.get("/projects")
async def get_projects(user_id: str, db=Depends(get_db)):
    service = ProjectService(db)
    return await service.list_projects(user_id)



@router.post("/conversations/{project_id}/message/")
async def send_message(
    project_id: str,
    body: MessageInput,
    user_id: str,
    db: Session = Depends(get_db),  # sync session
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    if body.content == "" or body.content is None or body.content.isspace():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")
    
    try:
        # Init chat service
        chat_service = ChatService(project_id=project_id, user_id=user_id, sql_db=db, mongo_db=mongo_db)
        chat_service.init()  # sync call

        # Create message
        message_data = MessageCreate(
            conversation_id=chat_service.conversation.id,
            content=body.content,
            type="HUMAN",
            sender_id=user_id
        )

        message_id = await chat_service.post_message(message_data)
        return {"message_id": message_id, "detail": "Message stored successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send message: {str(e)}"
        )

