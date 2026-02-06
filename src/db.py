"""Database connection for rpg-agents."""

import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# Global database instance
_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_db() -> AsyncIOMotorDatabase:
    """Connect to MongoDB and return the database."""
    global _client, _db
    
    if _db is not None:
        return _db
    
    mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "rpg_mcp")
    
    _client = AsyncIOMotorClient(mongo_url)
    _db = _client[db_name]
    
    # Create indexes
    await _db.users.create_index("email", unique=True)
    await _db.invite_codes.create_index("code", unique=True)
    await _db.world_access.create_index([("user_id", 1), ("world_id", 1)], unique=True)
    await _db.messages.create_index([("world_id", 1), ("created_at", -1)])
    
    return _db


async def get_db() -> AsyncIOMotorDatabase:
    """Get the database instance."""
    global _db
    if _db is None:
        return await connect_db()
    return _db


async def close_db():
    """Close the database connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
