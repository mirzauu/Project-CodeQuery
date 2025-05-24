from pydantic import BaseModel, EmailStr
from typing import Optional

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserLogin(UserBase):
    otp: str

class UserOut(UserBase):
    uid: str
    first_name: Optional[str]
    last_name: Optional[str]
