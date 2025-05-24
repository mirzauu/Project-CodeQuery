from typing import List, Optional
from pydantic import BaseModel


class ProviderInfo(BaseModel):
    id: str
    name: str
    description: str

class ModelInfo(BaseModel):
    provider: str
    id: str
    name: str