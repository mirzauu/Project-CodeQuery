import logging
from enum import Enum
from typing import List, Dict, Any, Union, AsyncGenerator, Optional
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import litellm
from litellm import litellm, AsyncOpenAI, acompletion
import instructor
import httpx
from pydantic import BaseModel

litellm.num_retries = 5  

class AgentProvider(Enum):
    CREWAI = "CREWAI"
    LANGCHAIN = "LANGCHAIN"
    PYDANTICAI = "PYDANTICAI"


AVAILABLE_MODELS = [
    AvailableModelOption(
        id="llama-3.3-70b-versatile",
        name="llama",
        description="OpenAI's latest model for complex tasks with large context",
        provider="llama",
    ),
]    



class ProviderService:
    def __init__(self, db, user_id: str):
        litellm.modify_params = True
        self.db = db
        self.llm = None
        self.user_id = user_id



    async def call_llm(
        self, messages: list, stream: bool = False, config_type: str = "chat"
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Call LLM with the specified messages."""
       

        # Build parameters using the config object
        params = self._build_llm_params(config)
        routing_provider = config.model.split("/")[0]

        # Get extra parameters and headers for API calls
        extra_params, _ = self.get_extra_params_and_headers(routing_provider)
        params.update(extra_params)

        # Handle streaming response if requested
        try:
            if stream:

                async def generator() -> AsyncGenerator[str, None]:
                    response = await acompletion(
                        messages=messages, stream=True, **params
                    )
                    async for chunk in response:
                        yield chunk.choices[0].delta.content or ""

                return generator()
            else:
                response = await acompletion(messages=messages, **params)
                return response.choices[0].message.content
        except Exception as e:
            logging.error(
                f"Error calling LLM: {e}, params: {params}, messages: {messages}"
            )
            raise e

    async def call_llm_with_structured_output(
        self, messages: list, output_schema: BaseModel, config_type: str = "chat"
    ) -> Any:
        """Call LLM and parse the response into a structured output using a Pydantic model."""
        # Select the appropriate config
        config = self.chat_config if config_type == "chat" else self.inference_config

        # Build parameters
        params = self._build_llm_params(config)
        routing_provider = config.model.split("/")[0]

        # Get extra parameters and headers
        extra_params, _ = self.get_extra_params_and_headers(routing_provider)

        try:
            if config.provider == "ollama":
                # use openai client to call ollama because of https://github.com/BerriAI/litellm/issues/7355
                client = instructor.from_openai(
                    AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
                    mode=instructor.Mode.JSON,
                )
                response = await client.chat.completions.create(
                    model=params["model"].split("/")[-1],
                    messages=messages,
                    response_model=output_schema,
                    temperature=params.get("temperature", 0.3),
                    max_tokens=params.get("max_tokens"),
                    **extra_params,
                )
            else:
                client = instructor.from_litellm(acompletion, mode=instructor.Mode.JSON)
                response = await client.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    response_model=output_schema,
                    strict=True,
                    temperature=params.get("temperature", 0.3),
                    max_tokens=params.get("max_tokens"),
                    api_key=params.get("api_key"),
                    **extra_params,
                )
            return response
        except Exception as e:
            logging.error(f"LLM call with structured output failed: {e}")
            raise e