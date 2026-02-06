"""Authentication endpoints and utilities."""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
import bcrypt
from bson import ObjectId

from ..db import get_db
from ..models import User, InviteCode

# Router
router = APIRouter(prefix="/auth", tags=["auth"])

# Security
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)  # For SSE endpoints that accept token as query param

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24 * 7  # 1 week


# Request/Response models
class RegisterRequest(BaseModel):
    """Registration request. Invite code required for account creation."""
    email: str
    password: str
    display_name: str
    invite_code: str  # Required - gates who can create accounts


class LoginRequest(BaseModel):
    """Login request."""
    email: str
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    """User info response."""
    id: str
    email: str
    display_name: str
    created_at: str


# Password utilities
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


# JWT utilities
def create_access_token(user_id: str, email: str) -> str:
    """Create a JWT access token."""
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode = {
        "sub": user_id,
        "email": email,
        "exp": expire,
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


# Dependency for authenticated routes
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get the current authenticated user."""
    token = credentials.credentials
    payload = decode_token(token)
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    db = await get_db()
    user_doc = await db.users.find_one({"_id": ObjectId(user_id)})
    
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    return User.from_doc(user_doc)


# Optional dependency for SSE endpoints that accept token as query param
async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_optional)
) -> User | None:
    """Get the current authenticated user, or None if no Authorization header."""
    if credentials is None:
        return None
    
    token = credentials.credentials
    payload = decode_token(token)
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    db = await get_db()
    user_doc = await db.users.find_one({"_id": ObjectId(user_id)})
    
    if not user_doc:
        return None
    
    return User.from_doc(user_doc)


# Endpoints
@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """
    Register a new user.
    
    Requires a valid invite code (for registration gating).
    After registration, user can create worlds or join existing ones with world codes.
    """
    db = await get_db()
    
    # Validate invite code (required for registration)
    invite_doc = await db.invite_codes.find_one({"code": request.invite_code})
    if not invite_doc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid invite code",
        )
    
    invite = InviteCode.from_doc(invite_doc)
    if not invite.is_valid():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invite code has expired or reached max uses",
        )
    
    # Check if email already exists
    existing = await db.users.find_one({"email": request.email.lower()})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Create user
    user = User(
        email=request.email.lower(),
        password_hash=hash_password(request.password),
        display_name=request.display_name,
    )
    
    result = await db.users.insert_one(user.to_doc())
    user.id = str(result.inserted_id)
    
    # Increment invite code usage
    try:
        invite_id = ObjectId(invite.id)
    except:
        invite_id = invite.id
    await db.invite_codes.update_one(
        {"_id": invite_id},
        {"$inc": {"uses": 1}}
    )
    
    # Generate token
    token = create_access_token(user.id, user.email)
    
    return TokenResponse(
        access_token=token,
        user=user.to_public(),
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Login and get an access token."""
    db = await get_db()
    
    # Find user
    user_doc = await db.users.find_one({"email": request.email.lower()})
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    user = User.from_doc(user_doc)
    
    # Verify password
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    # Generate token
    token = create_access_token(user.id, user.email)
    
    return TokenResponse(
        access_token=token,
        user=user.to_public(),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get the current user's info."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        created_at=current_user.created_at.isoformat(),
    )
