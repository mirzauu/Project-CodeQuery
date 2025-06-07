from typing import AsyncGenerator, Dict

from app.modules.provider.provider_service import ProviderService

from .agent_schema import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext
from .router_agent import RouterAgent



class ExecuterAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        agents: Dict[str, AgentConfig],
    ):
        self.agent = RouterAgent(llm_provider, agents=agents)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self.agent.run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunck in self.agent.run_stream(ctx):
            yield chunck
