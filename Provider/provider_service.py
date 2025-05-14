from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import litellm

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
        self.portkey_api_key = os.environ.get("PORTKEY_API_KEY", None)

        # Load user preferences
        user_pref = db.query(UserPreferences).filter_by(user_id=user_id).first()
        user_config = (
            user_pref.preferences if user_pref and user_pref.preferences else {}
        )

        # Create configurations based on user input (or fallback defaults)
        self.chat_config = build_llm_provider_config(user_config, config_type="chat")
        self.inference_config = build_llm_provider_config(
            user_config, config_type="inference"
        )