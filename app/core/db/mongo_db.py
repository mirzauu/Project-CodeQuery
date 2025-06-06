import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import config_provider

mongo_config = config_provider.get_mongo_config()

# Create a global MongoDB client and database reference
mongo_client = AsyncIOMotorClient(mongo_config["uri"])
db: AsyncIOMotorDatabase = mongo_client[mongo_config["db_name"]]
chat_collection = db["chat"]

# Dependency to be used in FastAPI
async def get_mongo_db() -> AsyncIOMotorDatabase:
    return db



