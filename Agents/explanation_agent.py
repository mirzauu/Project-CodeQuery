from .agent_schema import ChatAgent, ChatAgentResponse, ChatContext,AgentConfig,TaskConfig
from .framework.pydantic_agent import PydanticRagAgent

from typing import AsyncGenerator

class ExplanationAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
    ):
        self.tools_provider = tools_provider
        self.llm_provider = llm_provider

    def _build_agent(self):
        agent_config = AgentConfig(
            role="Code Explanation Assistant",
            goal="Explain code functionality, structure, and logic clearly",
            backstory="You are an AI assistant skilled at explaining source code in simple, understandable terms for developers of all levels.",
            tasks=[
                TaskConfig(
                    description=explanation_task_prompt,
                    expected_output="Clear explanation of what the code does, how it works, and any relevant context or dependencies.",
                )
            ],
        )
        tools = self.tools_provider.get_tools(
            [
                "get_nodes_from_tags",
                "ask_knowledge_graph_queries",
                "get_code_from_multiple_node_ids",
                "web_search_tool",
                "github_tool",
            ]
        )

        return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


explanation_task_prompt = """
You are provided with code snippets, modules, or file diffs from a codebase.

Your goal is to explain the functionality, structure, and purpose of the given code in a clear and beginner-friendly manner. You can use the knowledge graph tool to answer questions or provide context about the codebase if needed.

Follow these steps to construct your explanation:

1. **Understand the Code Context**:
   - Identify the main components such as functions, classes, methods, or logic blocks.
   - Determine the purpose of each component and how they work together.

2. **Use the Knowledge Graph Tool (if required)**:
   - If a specific question is asked, or if more context is needed, formulate a query.
   - Focus on key identifiers, docstring phrases, technical keywords, and their relationships.
   - Use keyword-based search terms like "initializes database", "sends HTTP request", "returns serialized response".

3. **Break Down and Explain**:
   - Describe what the code does at a high level.
   - Explain each function or class in simple terms.
   - Highlight key logic, algorithms, or design patterns.
   - Clarify any advanced or non-obvious behavior in the code.

4. **Use Examples and Analogies (if applicable)**:
   - Add examples or analogies to clarify complex parts.
   - Explain edge cases, assumptions, or parameter behavior.

5. **Respond to the User's Query**:
   - Consider the user's specific question: {query}
   - Tailor your explanation to address their concern or area of interest.
   - Be clear, concise, and avoid jargon unless necessary.

6. **Structure Your Output**:
   - Start with a summary of the overall code functionality.
   - Follow with detailed breakdowns of each relevant part.
   - End with clarification of how everything fits together and any best practices or suggestions.

The goal is to make the code understandable to someone unfamiliar with it or still learning.
"""
