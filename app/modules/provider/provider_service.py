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

import json
import logging
import os
from typing import Any
from pydantic import BaseModel, ValidationError
from mistralai import Mistral
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
    def __init__(self, db):
        litellm.modify_params = True
        self.db = db
        self.llm = None



    async def call_llm(
        self, messages: list, stream: bool = False, config_type: str = "chat"
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Call LLM with the specified messages."""
        pass

        

    async def call_llm_with_structured_output(
    self, messages: list, output_schema: type[BaseModel], config_type: str = "chat"
) -> Any:
            """Call LLM and parse the response into a structured output using a Pydantic model."""
            print("REACHING CALL LLM WITH STRUCTURED OUTPUT")
            api_key = "I8dVoJSO5XmpMUcyIQ0KRiGNfduJRCM8"
            model = "mistral-large-latest"
            client = Mistral(api_key=api_key)

            try:
                # 🧠 Step 1: Make the LLM request
                response = client.chat.complete(
                    model=model,
                    messages=messages,
                    response_format= {
                        "type": "json_object",
                    },  # ✅ Should be a string
                )

                raw = response.choices[0].message.content
                logging.info(f"Raw LLM response: {raw}")
                # 🧠 Step 2: Try parsing the response content
                try:
                    parsed_data = json.loads(raw)

                    if isinstance(parsed_data, list):
                        parsed_data = {"docstrings": parsed_data}

                except json.JSONDecodeError as json_err:
                    logging.warning(f"Failed to parse JSON from response: {json_err}")
                    # ⛑️ Fallback: Return the raw string or log and exit
                    return {"error": "Failed to parse JSON", "raw_response": response.choices[0].message.content}

                # 🧠 Step 3: Try validating the schema
                try:
                    validated_output = output_schema.model_validate(parsed_data)
                    return validated_output

                except ValidationError as val_err:
                    logging.warning(f"Pydantic validation failed: {val_err}")
                    # ⛑️ Fallback: Return raw parsed data with errors
                    return {"error": "Validation failed", "parsed_data": parsed_data, "validation_errors": val_err.errors()}

            except Exception as e:
                logging.error(f"LLM call failed. Messages: {messages}, Error: {e}")
                return {"error": "LLM call failed", "exception": str(e)}
        