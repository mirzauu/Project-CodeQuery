import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(override=True)

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["tfo"]  # Your MongoDB database
chat_collection = db["chat"]  # Collection for chat messages


