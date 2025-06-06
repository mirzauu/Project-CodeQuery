from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import HTTPException

from app.core.config import config_provider

jwt_config = config_provider.get_jwt_config()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=jwt_config["access_token_expire_minutes"]))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, jwt_config["secret_key"], algorithm=jwt_config["algorithm"])

def verify_token(token: str):
    try:
        payload = jwt.decode(token, jwt_config["secret_key"], algorithms=[jwt_config["algorithm"]])
        return payload
    except JWTError:
        return None

def decode_jwt_token(token: str):
    try:
        payload = jwt.decode(token, jwt_config["secret_key"], algorithms=[jwt_config["algorithm"]])
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(tz=timezone.utc):
            raise HTTPException(status_code=401, detail="Token expired")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
