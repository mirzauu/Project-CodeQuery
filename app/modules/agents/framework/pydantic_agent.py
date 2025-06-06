import re
from typing import List, AsyncGenerator
import logging

from app.modules.tools.tool_helper import (
    get_tool_call_info_content,
    get_tool_response_message,
    get_tool_result_info_content,
    get_tool_run_message,
)
from app.modules.provider.provider_service import (
    ProviderService,
)
from ..agent_schema import ChatAgent, ChatAgentResponse, ChatContext,AgentConfig,TaskConfig, ToolCallEventType,ToolCallResponse




from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    ModelResponse,
    TextPart,
)
from langchain_core.tools import StructuredTool

from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider 

logger = logging.getLogger(__name__)


    
class PydanticRagAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
    ):
        """Initialize the agent with configuration and tools"""

        self.tasks = config.tasks
        self.max_iter = config.max_iter

        # tool name can't have spaces for langgraph/pydantic agents
        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        self.agent = Agent(
            model=MistralModel(
                model_name="mistral-large-latest",
                provider=MistralProvider(api_key="I8dVoJSO5XmpMUcyIQ0KRiGNfduJRCM8")
            ),
            tools=[
                Tool(
                    name=tool.name,
                    description=tool.description,
                    function=tool.func,  # type: ignore
                )
                for tool in tools
            ],
            system_prompt=f"Role: {config.role}\nGoal: {config.goal}\nBackstory: {config.backstory}. Respond to the user query",
            result_type=str,
            retries=3,
            defer_model_check=True,
            end_strategy="exhaustive",
            model_settings={"parallel_tool_calls": True, "max_tokens": 8000},
        )

    
    def _create_task_description(
        self,
        task_config: TaskConfig,
        ctx: ChatContext,
    ) -> str:
        """Create a task description from task configuration"""
        

        return f"""
                CONTEXT:
                User Query: {ctx.query}
                Project ID: {ctx.project_id}
                
                Additional Context:
                {ctx.additional_context if ctx.additional_context != "" else "no additional context"}

                TASK:
                {task_config.description}

                Expected Output:
                {task_config.expected_output}

                INSTRUCTIONS:
                1. Use the available tools to gather information
                2. Process and synthesize the gathered information
                3. Format your response in markdown, make sure it's well formatted
                4. Include relevant code snippets and file references
                5. Provide clear explanations
                6. Verify your output before submitting

                IMPORTANT:
                - Use tools efficiently and avoid unnecessary API calls
                - Only use the tools listed below

                With above information answer the user query: {ctx.query}
            """


    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow"""
        logger.info("running pydantic-ai agent")
        try:

            task = self._create_task_description(self.tasks[0], ctx)

            resp = await self.agent.run(user_prompt=task)

            return ChatAgentResponse(
                response=resp.data,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        print("running pydantic-ai agent stream",ctx)
        task = self._create_task_description(self.tasks[0], ctx)
        print(f"Task description: {task}")
        try:
            async with self.agent.iter(
                user_prompt=task,
                message_history=[
                    ModelResponse([TextPart(content=msg)]) for msg in ctx.history
                ],
            ) as run:
                print("run started")
                async for node in run:
                    if Agent.is_model_request_node(node):
                        # A model request node => We can stream tokens from the model's request
                        print("model request node found")
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if isinstance(event, PartStartEvent) and isinstance(
                                    event.part, TextPart
                                ):
                                    yield ChatAgentResponse(
                                        response=event.part.content,
                                        tool_calls=[],
                                        citations=[],
                                    )
                               
                                if isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    yield ChatAgentResponse(
                                        response=event.delta.content_delta,
                                        tool_calls=[],
                                        citations=[],
                                    )

                    elif Agent.is_call_tools_node(node):
                        print("call tools node found")
                        async with node.stream(run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    summary = f"Calling `{event.part.tool_name}` with args: {event.part.args_as_dict()}"
                                    tool_call = ToolCallResponse(
                                        call_id=event.part.tool_call_id or "",
                                        event_type="call",
                                        tool_name=event.part.tool_name,
                                        tool_response="Running tool...",
                                        tool_call_details={"summary": summary},
                                    )

                                    yield ChatAgentResponse(
                                        response="",
                                        tool_calls=[tool_call],
                                        citations=[],
                                    )
                            
                                if isinstance(event, FunctionToolResultEvent):
                                    summary = f"Calling `{event.result.tool_name}` with args: {event.result.content}"
                                    tool_call=ToolCallResponse(
                                        call_id=event.result.tool_call_id or "",
                                        event_type="call",
                                        tool_name=event.result.tool_name,
                                        tool_response="Running tool...",
                                        tool_call_details={"summary": summary},
                                    )

                                    yield ChatAgentResponse(
                                        response="",
                                        tool_calls=[tool_call],
                                        citations=[],
                                    )

                    elif Agent.is_end_node(node):
                        logger.info("result streamed successfully!!")
                        print("end node found")

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e

