import random
import string
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from datetime import timezone
from .user_model import User
from app.core.security import create_access_token
from .user_schema import UserCreate, UserLogin
from fastapi import HTTPException, status
from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.core.security import decode_jwt_token
from app.core.db.postgress_db import get_db

from app.core.email_utils import send_email_otp


class UserService:
    OTP_EXPIRY_MINUTES = 10

    def __init__(self, db: Session):
        self.db = db

    # ---------------------------
    # Generate Random OTP Code
    # ---------------------------
    def _generate_otp(self, length=6):
        return ''.join(random.choices(string.digits, k=length))

    # ---------------------------
    # Generate UID
    # ---------------------------
    def _generate_uid(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=24))
    

    def signup(self, user_data: UserCreate):
        user = self.db.query(User).filter_by(email=user_data.email).first()
        otp_code = self._generate_otp()
        expiry_time = datetime.utcnow() + timedelta(minutes=self.OTP_EXPIRY_MINUTES)

        if user:
            user.otp_code = otp_code
            user.otp_expiry = expiry_time
            user.first_name = user_data.first_name or user.first_name
            user.last_name = user_data.last_name or user.last_name
        else:
            user = User(
                uid=self._generate_uid(),
                email=user_data.email,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                otp_code=otp_code,
                otp_expiry=expiry_time,
            )
            self.db.add(user)

        self.db.commit()

        # Send OTP to user's email
        send_email_otp(user.email, otp_code)

        return {"message": "OTP sent successfully to email."}

    # ---------------------------
    # Login: Verify OTP, Return Token
    # ---------------------------
    def login(self, login_data: UserLogin):
        user = self.db.query(User).filter_by(email=login_data.email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.otp_code != login_data.otp:
            raise HTTPException(status_code=400, detail="Invalid OTP")
        if not user.otp_expiry or user.otp_expiry < datetime.utcnow():
            raise HTTPException(status_code=400, detail="OTP expired")

        user.email_verified = True
        user.last_login_at = datetime.utcnow()
        user.otp_code = None
        user.otp_expiry = None
        self.db.commit()

        token = create_access_token({"uid": user.uid})
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "uid": user.uid,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
            },
        }

    # ---------------------------
    # Send OTP to Email
    # ---------------------------
    def send_otp(self, user_data: UserCreate):
        user = self.db.query(User).filter_by(email=user_data.email).first()
        otp_code = self._generate_otp()
        expiry_time = datetime.utcnow() + timedelta(minutes=self.OTP_EXPIRY_MINUTES)

        if user:
            user.otp_code = otp_code
            user.otp_expiry = expiry_time
            if user_data.first_name:
                user.first_name = user_data.first_name
            if user_data.last_name:
                user.last_name = user_data.last_name
        else:
            user = User(
                uid=self._generate_uid(),
                email=user_data.email,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                otp_code=otp_code,
                otp_expiry=expiry_time
            )
            self.db.add(user)

        self.db.commit()

        # ✅ Send email via SMTP
        send_email_otp(user.email, otp_code)

        return {"message": "OTP sent successfully to email."}

    # ---------------------------
    # Verify OTP and Login
    # ---------------------------
    def verify_otp_and_login(self, login_data: UserLogin):
        user = self.db.query(User).filter_by(email=login_data.email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.otp_code != login_data.otp:
            raise HTTPException(status_code=400, detail="Invalid OTP")
        # if user.otp_expiry is None or user.otp_expiry < datetime.now(timezone.utc):
        #     raise HTTPException(status_code=400, detail="OTP expired")

        # Update user status
        user.email_verified = True
        user.last_login_at = datetime.utcnow()
        # user.otp_code = None
        # user.otp_expiry = None
        self.db.commit()

        token = create_access_token({"sub": user.uid})
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "uid": user.uid,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
            }
        }

    # ---------------------------
    # Get User By UID
    # ---------------------------
    def get_user_by_uid(self, uid: str):
        user = self.db.query(User).filter_by(uid=uid).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    # ---------------------------
    # Get Projects for User
    # ---------------------------
    def get_user_projects(self, uid: str):
        user = self.get_user_by_uid(uid)
        return user.projects if hasattr(user, "projects") else []
    

    def check_auth(
        request: Request,
        res: Response,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
        db: Session = Depends(get_db),
    ):
        if credentials is None:
            raise HTTPException(status_code=401, detail="Missing authorization token")

        token = credentials.credentials
        payload = decode_jwt_token(token)
        user_id = payload.get("sub")
        print("user detials",payload,user_id)

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        user = db.query(User).filter_by(uid=user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        request.state.user = user  # optional: attach user to request
        return user
