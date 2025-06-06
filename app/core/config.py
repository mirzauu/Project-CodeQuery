import os
from dotenv import load_dotenv

load_dotenv()


class ConfigProvider:
    def __init__(self):
        self.neo4j_config = {
            "uri": os.getenv("NEO4J_URI"),
            "username": os.getenv("NEO4J_USERNAME"),
            "password": os.getenv("NEO4J_PASSWORD"),
        }

    def get_neo4j_config(self):
        return self.neo4j_config

    def get_llm_api_key(self):
        return os.getenv("LLM_API_KEY", "your_default_llm_api_key")

    def get_smtp_config(self):
        smtp_user = os.getenv("SMTP_USER")
        return {
            "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "port": int(os.getenv("SMTP_PORT", 587)),
            "user": smtp_user,
            "password": os.getenv("SMTP_PASSWORD"),
            "email_from": os.getenv("EMAIL_FROM", smtp_user),
        }

    def get_jwt_config(self):
        return {
            "secret_key": os.getenv("JWT_SECRET", "your-secret-key"),
            "algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
            "access_token_expire_minutes": int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 300)),
        }
    
    def get_mongo_config(self):
        return {
            "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            "db_name": os.getenv("MONGO_DB_NAME", "tfo"),
        }




config_provider = ConfigProvider()
