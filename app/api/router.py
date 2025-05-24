from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from schemas.user import UserCreate, UserLogin
from services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/send-otp")
def send_otp(data: UserCreate, db: Session = Depends(get_db)):
    service = UserService(db)
    return service.send_otp(data.email, data.first_name, data.last_name)

@router.post("/verify-otp")
def verify_otp(data: UserLogin, db: Session = Depends(get_db)):
    service = UserService(db)
    try:
        return service.verify_otp_and_login(data.email, data.otp)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
