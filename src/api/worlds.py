"""World endpoints - list, get, create, join, world codes."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel
from bson import ObjectId

from ..db import get_db
from ..models import User, WorldAccess, WorldCode, Message
from .auth import get_current_user

# Router
router = APIRouter(prefix="/worlds", tags=["worlds"])


# Response models
class WorldSummary(BaseModel):
    """World summary for list view."""
    id: str
    name: str
    description: str
    role: str
    character_id: Optional[str]
    character_name: Optional[str]


class WorldDetail(BaseModel):
    """Full world details."""
    id: str
    name: str
    description: str
    settings: dict
    game_time: int
    role: str
    character: Optional[dict]


class MessageResponse(BaseModel):
    """Message for API response."""
    id: str
    world_id: str
    user_id: Optional[str]
    character_name: str
    content: str
    message_type: str
    created_at: str


class MessagesResponse(BaseModel):
    """Paginated messages response."""
    messages: list[MessageResponse]
    has_more: bool


class WorldCodeResponse(BaseModel):
    """Created world code response."""
    code: str
    world_id: str
    expires_at: Optional[str]
    max_uses: Optional[int]


class CreateWorldCodeRequest(BaseModel):
    """Request to create a world code for inviting players."""
    max_uses: Optional[int] = None
    expires_in_hours: Optional[int] = 24 * 7  # Default 1 week


class JoinWorldRequest(BaseModel):
    """Request to join a world using a world code."""
    code: str


class JoinWorldResponse(BaseModel):
    """Response after joining a world."""
    world_id: str
    world_name: str
    role: str


class CreateWorldRequest(BaseModel):
    """Request to create a new world."""
    name: str


class CreateWorldResponse(BaseModel):
    """Response after creating a world."""
    id: str
    name: str
    description: str
    role: str


# Endpoints
@router.get("", response_model=list[WorldSummary])
async def list_worlds(current_user: User = Depends(get_current_user)):
    """List all worlds the user has access to."""
    db = await get_db()
    
    # Get all world access records for user
    access_cursor = db.world_access.find({"user_id": current_user.id})
    access_list = await access_cursor.to_list(length=100)
    
    worlds = []
    for access_doc in access_list:
        access = WorldAccess.from_doc(access_doc)
        
        # Get world info - handle both ObjectId and string IDs
        try:
            world_id = ObjectId(access.world_id)
        except:
            world_id = access.world_id
        world_doc = await db.worlds.find_one({"_id": world_id})
        if not world_doc:
            continue
        
        # Get character name if assigned
        character_name = None
        if access.character_id:
            char_doc = await db.characters.find_one({"_id": ObjectId(access.character_id)})
            if char_doc:
                character_name = char_doc.get("name")
        
        worlds.append(WorldSummary(
            id=str(world_doc["_id"]),
            name=world_doc.get("name", ""),
            description=world_doc.get("description", ""),
            role=access.role,
            character_id=access.character_id,
            character_name=character_name,
        ))
    
    return worlds


@router.post("", response_model=CreateWorldResponse)
async def create_world(
    request: CreateWorldRequest,
    current_user: User = Depends(get_current_user),
):
    """Create a new empty world. The creator becomes a 'god' of the world."""
    db = await get_db()
    
    # Create the world document
    world_doc = {
        "name": request.name,
        "description": "",  # Empty, to be filled by GM agent
        "settings": {},
        "game_time": 0,
        "created_at": datetime.utcnow(),
        "created_by": current_user.id,
    }
    
    result = await db.worlds.insert_one(world_doc)
    world_id = str(result.inserted_id)
    
    # Create world access - creator is a god
    world_access = WorldAccess(
        user_id=current_user.id,
        world_id=world_id,
        role="god",
    )
    await db.world_access.insert_one(world_access.to_doc())
    
    return CreateWorldResponse(
        id=world_id,
        name=request.name,
        description="",
        role="god",
    )


@router.get("/{world_id}", response_model=WorldDetail)
async def get_world(world_id: str, current_user: User = Depends(get_current_user)):
    """Get details for a specific world."""
    db = await get_db()
    
    # Check access
    access_doc = await db.world_access.find_one({
        "user_id": current_user.id,
        "world_id": world_id,
    })
    
    if not access_doc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this world",
        )
    
    access = WorldAccess.from_doc(access_doc)
    
    # Get world - handle both ObjectId and string IDs
    try:
        world_id_query = ObjectId(world_id)
    except:
        world_id_query = world_id
    world_doc = await db.worlds.find_one({"_id": world_id_query})
    if not world_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="World not found",
        )
    
    # Get character if assigned
    character = None
    if access.character_id:
        char_doc = await db.characters.find_one({"_id": ObjectId(access.character_id)})
        if char_doc:
            char_doc["_id"] = str(char_doc["_id"])
            character = char_doc
    
    return WorldDetail(
        id=str(world_doc["_id"]),
        name=world_doc.get("name", ""),
        description=world_doc.get("description", ""),
        settings=world_doc.get("settings", {}),
        game_time=world_doc.get("game_time", 0),
        role=access.role,
        character=character,
    )


# DEPRECATED: Message endpoints moved to messages module
# @router.get("/{world_id}/messages", response_model=MessagesResponse)
# async def get_messages(
#     world_id: str,
#     limit: int = Query(default=50, le=100),
#     before: Optional[str] = Query(default=None, description="Message ID to get messages before"),
#     current_user: User = Depends(get_current_user),
# ):
#     """Get chat messages for a world (paginated)."""
#     # ... endpoint moved to messages.py ...


@router.post("/{world_id}/code", response_model=WorldCodeResponse)
async def create_world_code(
    world_id: str,
    request: CreateWorldCodeRequest,
    current_user: User = Depends(get_current_user),
):
    """Create a world code for inviting players (gods only)."""
    db = await get_db()
    
    # Check access and role
    access_doc = await db.world_access.find_one({
        "user_id": current_user.id,
        "world_id": world_id,
    })
    
    if not access_doc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this world",
        )
    
    access = WorldAccess.from_doc(access_doc)
    
    if access.role != "god":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only gods can create world codes",
        )
    
    # Calculate expiration
    expires_at = None
    if request.expires_in_hours:
        expires_at = datetime.utcnow() + timedelta(hours=request.expires_in_hours)
    
    # Create world code
    world_code = WorldCode(
        world_id=world_id,
        created_by=current_user.id,
        max_uses=request.max_uses,
        expires_at=expires_at,
    )
    
    result = await db.world_codes.insert_one(world_code.to_doc())
    world_code.id = str(result.inserted_id)
    
    return WorldCodeResponse(
        code=world_code.code,
        world_id=world_code.world_id,
        expires_at=world_code.expires_at.isoformat() if world_code.expires_at else None,
        max_uses=world_code.max_uses,
    )


@router.post("/join", response_model=JoinWorldResponse)
async def join_world(
    request: JoinWorldRequest,
    current_user: User = Depends(get_current_user),
):
    """Join a world using a world code."""
    db = await get_db()
    
    # Find the world code
    code_doc = await db.world_codes.find_one({"code": request.code})
    if not code_doc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid world code",
        )
    
    world_code = WorldCode.from_doc(code_doc)
    if not world_code.is_valid():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="World code has expired or reached max uses",
        )
    
    # Check if user already has access to this world
    existing_access = await db.world_access.find_one({
        "user_id": current_user.id,
        "world_id": world_code.world_id,
    })
    
    if existing_access:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already have access to this world",
        )
    
    # Get world info
    try:
        world_id_query = ObjectId(world_code.world_id)
    except:
        world_id_query = world_code.world_id
    world_doc = await db.worlds.find_one({"_id": world_id_query})
    if not world_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="World not found",
        )
    
    # Create world access as mortal
    world_access = WorldAccess(
        user_id=current_user.id,
        world_id=world_code.world_id,
        role="mortal",
    )
    await db.world_access.insert_one(world_access.to_doc())
    
    # Increment world code usage
    try:
        code_id = ObjectId(world_code.id)
    except:
        code_id = world_code.id
    await db.world_codes.update_one(
        {"_id": code_id},
        {"$inc": {"uses": 1}}
    )
    
    return JoinWorldResponse(
        world_id=world_code.world_id,
        world_name=world_doc.get("name", ""),
        role="mortal",
    )


@router.patch("/{world_id}/character")
async def set_character(
    world_id: str,
    character_id: str,
    current_user: User = Depends(get_current_user),
):
    """Set the user's character for a world."""
    db = await get_db()
    
    # Check access
    access_doc = await db.world_access.find_one({
        "user_id": current_user.id,
        "world_id": world_id,
    })
    
    if not access_doc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this world",
        )
    
    # Verify character exists and belongs to this world
    char_doc = await db.characters.find_one({
        "_id": ObjectId(character_id),
        "world_id": world_id,
        "is_player_character": True,
    })
    
    if not char_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found or not a player character",
        )
    
    # Update world access
    await db.world_access.update_one(
        {"user_id": current_user.id, "world_id": world_id},
        {"$set": {"character_id": character_id}}
    )
    
    return {"status": "ok", "character_id": character_id}
