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

from app.core.config import config_provider

litellm.num_retries = 5  

class AgentProvider(Enum):
    CREWAI = "CREWAI"
    LANGCHAIN = "LANGCHAIN"
    PYDANTICAI = "PYDANTICAI"


# AVAILABLE_MODELS = [
#     AvailableModelOption(
#         id="llama-3.3-70b-versatile",
#         name="llama",
#         description="OpenAI's latest model for complex tasks with large context",
#         provider="llama",
#     ),
# ]    



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
        pass

        

    async def call_llm_with_structured_output(
        self, messages: list, output_schema: type[BaseModel], config_type: str = "chat"
    ) -> Any:
        """Call LLM and parse the response into a structured output using a Pydantic model."""
        # Select the appropriate config
        import json
        from mistralai import Mistral

        api_key = config_provider.get_llm_api_key()
        if not api_key:
            raise ValueError("LLM API key is not set in the configuration.")
        
        model = "mistral-large-latest"

        client = Mistral(api_key=api_key)

# Initialize the client with your API key
        

        try:
            # Make the LLM request
            response = client.chat.complete(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )

            # Get the JSON string
            raw = response.choices[0].message.content
            logging.info(f"Raw LLM response: {raw}")

            # Parse the JSON string into Python list
            parsed_data = json.loads(raw)

            # ✅ Ensure it's wrapped in a dict with "docstrings" key
            if isinstance(parsed_data, list):
                parsed_data = {"docstrings": parsed_data}

            # ✅ Now validate the parsed_data against the Pydantic model
            validated_output = output_schema.model_validate(parsed_data)
            return validated_output

        except Exception as e:
            logging.error(f"LLM call with structured output failed: {e}")
            raise e