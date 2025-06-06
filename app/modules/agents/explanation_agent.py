from .agent_schema import ChatAgent, ChatAgentResponse, ChatContext,AgentConfig,TaskConfig
from .framework.pydantic_agent import PydanticRagAgent

from typing import AsyncGenerator
from app.modules.provider.provider_service import (
    ProviderService,
)

class ExplanationAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,

    ):
        self.llm_provider = llm_provider

    def _build_agent(self):
        agent_config = AgentConfig(
            role="Blog Writing Assistant",
            goal="Write clear, structured, and insightful blog content that explains the functionality, structure, and logic of source code.",
            backstory="You are an AI writing assistant specialized in creating developer-friendly blog posts. Your strength lies in breaking down complex code into simple, understandable explanations with helpful examples, context, and technical clarity.",
            tasks=[
                TaskConfig(
                    description="Write a blog-style explanation of the provided code. Break down what the code does, how it works, and include any important details such as dependencies, use cases, or improvements.",
                    expected_output="A well-structured blog post that explains the code clearly, using headings, bullet points, and code snippets as needed. The tone should be educational, engaging, and suitable for readers ranging from beginner to intermediate developers."
                )
            ]

        )
        tools = [
 
                "extract_keywords",
              
            ]
        

        return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        print(f"Running ExplanationAgent with context: {ctx}")
        return await self._build_agent().run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        print(f"Running ExplanationAgent in stream mode with context: {ctx}")
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk

explanation_task_prompt = """
You are provided with one or more code snippets, modules, or file diffs from a project.

Your task is to write a clear, well-structured blog post that explains the functionality, structure, and purpose of the code in a way that is accessible to developers of all levels—especially beginners and intermediates.

Follow this approach when crafting your post:

1. **Start with a Brief Introduction**:
   - Give a short overview of the topic or feature the code addresses.
   - Mention why this code matters or where it fits in a real-world project.

2. **Understand and Analyze the Code**:
   - Identify key components such as functions, classes, logic blocks, and patterns.
   - Explain their roles and how they interact within the code.

3. **Break It Down Section by Section**:
   - Walk through each part of the code in a logical sequence.
   - Use simple language to describe what each part does.
   - Include inline or block-style code snippets for reference.
   - Highlight any unique design patterns, algorithms, or workflows.

4. **Use Examples, Analogies, and Diagrams (if helpful)**:
   - Add real-world analogies or visual metaphors to clarify tricky parts.
   - Mention any edge cases, assumptions, or parameter behavior worth noting.

5. **Add Technical Context**:
   - Mention dependencies, libraries, or frameworks used.
   - Point out common gotchas, alternative approaches, or best practices.

6. **Answer the Reader’s Intent**:
   - If the explanation responds to a specific question or concept, keep the focus clear.
   - Make sure the reader walks away with actionable understanding.

7. **Structure Your Blog Post for Readability**:
   - Use headings, subheadings, bullet points, and code blocks.
   - Start with a summary, explain in detail, and end with a recap or takeaway.

Your goal is to turn code into a readable, friendly blog article that educates, engages, and empowers developers at any level to understand and use the code confidently.
"""
