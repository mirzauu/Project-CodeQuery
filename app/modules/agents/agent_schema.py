from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task

class ToolCallEventType(Enum):
    CALL = "call"
    RESULT = "result"


class ToolCallResponse(BaseModel):
    call_id: str = Field(
        ...,
        description="ID of the tool call",
    )
    event_type: ToolCallEventType = Field(..., description="Type of the event")
    tool_name: str = Field(
        ...,
        description="Name of the tool",
    )
    tool_response: str = Field(
        ...,
        description="Response from the tool",
    )
    tool_call_details: Dict[str, Any] = Field(
        ...,
        description="Details of the tool call",
    )


class ChatAgentResponse(BaseModel):
    response: str = Field(
        ...,
        description="Full response to the query",
    )
    tool_calls: List[ToolCallResponse] = Field([], description="List of tool calls")
    citations: List[str] = Field(
        ...,
        description="List of file names extracted from context and referenced in the response",
    )

class ChatContext(BaseModel):
    project_id: str
    history: List[str]
    query: str
    additional_context: str = ""


class ChatAgent(ABC):
    """Interface for chat agents. Chat agents will be used in conversation APIs"""

    @abstractmethod
    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Run synchronously in a blocking manner, return entire response at once"""
        pass

    @abstractmethod
    def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        """Run asynchronously, yield response piece by piece"""
        pass


class TaskConfig(BaseModel):
    """Model for task configuration from agent_config.json"""

    description: str
    expected_output: str
    context: Optional[Task] = None


class AgentConfig(BaseModel):
    """Model for agent configuration from agent_config.json"""

    role: str
    goal: str
    backstory: str
    tasks: List[TaskConfig]
    max_iter: int = 15

class AgentInfo(BaseModel):
    id: str
    name: str
    description: str
    agent: str


class AgentWithInfo:
    def __init__(self, agent: ChatAgent, id: str, name: str, description: str):
        self.id = id
        self.name = name
        self.description = description
        self.agent = agent
