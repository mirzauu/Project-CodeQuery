import os
from typing import List, Optional


from .agent_schema import AgentWithInfo, ChatContext
from .explanation_agent import ExplanationAgent
from .qna_agent import QnAAgent
from app.modules.provider.provider_service import ProviderService
from app.modules.tools.tool_service import ToolService

from app.modules.agents.executer_agent import (
    ExecuterAgent,
)
from pydantic import BaseModel



class AgentsService:
    def __init__(
        self,
        db,
        user_id: str,
        llm_provider: ProviderService
    ):  
        self.db = db
        self.user_id= user_id
        self.tool=ToolService(db,user_id)
        self.system_agents = self._system_agents(
            llm_provider,self.tool
        )
        self.executer = ExecuterAgent(llm_provider, self.system_agents)
        
    def _system_agents(
        self,
        llm_provider: ProviderService,
        tool=ToolService
    ):
        return {
            "explanation_agent": AgentWithInfo(
                id="explanation_agent",
                name="Explantion Agent",
                description="An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
                agent=ExplanationAgent(llm_provider),
            ),
            "qna_agent": AgentWithInfo(
                id="qna_agent",
                name="qna Agent",
                description="An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
                agent=QnAAgent(llm_provider,tool),
            ),
        }

    async def execute(self, ctx: ChatContext):
        return await self.executer.run(ctx)

    async def execute_stream(self, ctx: ChatContext):
        async for chunk in self.executer.run_stream(ctx):
            yield chunk

   