from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.db.postgress_db import get_db


from app.modules.users.user_schema import UserCreate, UserLogin
from app.modules.knowledge.schema.parsing_schema import ParseRequest
from app.modules.users.user_service import UserService
from app.modules.conversation.project.project_service import ProjectService
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
    print(data.repo_link,user.uid)
    return await parse.parse_validator(data.repo_link,user)


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