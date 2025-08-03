from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, Security, Request
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from src import config

# API Key security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

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

async def verify_api_key(
    request: Request,
    header_key: str = Security(api_key_header),
    query_key: str = Security(api_key_query)
):
    """
    Verify API key for protected endpoints.
    Checks for API key in multiple places:
    1. X-API-Key header
    2. api_key query parameter
    3. Authorization header (Bearer token)
    """
    api_key = None
    
    # Check API key in X-API-Key header
    if header_key:
        api_key = header_key
    
    # Check API key in query parameter
    elif query_key:
        api_key = query_key
        
    # Check API key in Authorization header
    else:
        auth_header = request.headers.get("Authorization")
        if auth_header:
            scheme, token = get_authorization_scheme_param(auth_header)
            if scheme.lower() == "bearer":
                api_key = token
    
    # Validate the API key
    if not api_key:
        from src.logging_config import api_logger
        api_logger.warning(
            "API key missing in request. "
            f"Headers: {dict(request.headers)} "
            f"Query params: {dict(request.query_params)}"
        )
        raise HTTPException(
            status_code=401,
            detail="API key is missing. Please provide it via X-API-Key header, api_key query parameter, or Authorization Bearer token."
        )
    
    if api_key != config.API_KEY:
        from src.logging_config import api_logger
        # Log first 4 chars of provided key for debugging (don't log full key for security)
        masked_key = api_key[:4] + "*" * (len(api_key) - 4) if len(api_key) > 4 else "****"
        api_logger.warning(f"Invalid API key provided: {masked_key}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Log successful authentication
    from src.logging_config import api_logger
    api_logger.info(f"API key validated for endpoint: {request.url.path}")
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
