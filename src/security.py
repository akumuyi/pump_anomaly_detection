from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from src import config

# API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# JWT token security
SECRET_KEY = config.SECRET_KEY
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for protected endpoints."""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is missing"
        )
    if api_key != config.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key

def verify_token(token: str):
    """Verify JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )
