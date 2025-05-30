import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

load_dotenv(override=True)

MONGO_URI = "mongodb+srv://ainypus:3mz1b0dZcWKPYxtZ@cluster0.ksi9c62.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a global client
mongo_client = AsyncIOMotorClient(MONGO_URI)
db: AsyncIOMotorDatabase = mongo_client["tfo"]  # Async database
chat_collection = db["chat"]  # Async collection

# Dependency to be used in FastAPI
async def get_mongo_db() -> AsyncIOMotorDatabase:
    return db
