# file: config.py
from dotenv import load_dotenv
load_dotenv()
from pydantic_settings import BaseSettings
from pydantic import ConfigDict 



            # ConfigDict here
class AppConfig(BaseSettings):
    """
    Centralized configuration management using Pydantic.
    Automatically reads environment variables, validates types,
    and applies default values if missing.
    """

    # API Keys
    OPENAI_API_KEY: str
    GEMINAI_API_KEY: str
    GROQ_API_KEY:str
    OPENROUTER_API_KEY: str
    GEMINAI_API_KEY2:str
    PINECONE_API_KEY:str
    COHERE_API:str

    # Model config
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    GROQ_MODEL:str="openai/gpt-oss-120b"
    MIMO_MODEL: str = "xiaomi/mimo-v2-flash:free"


    model_config=ConfigDict(
        # Automatically load environment variables from a .env file
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore"   ## pydantic will throw error if we have mention other Variable  other then mention here , 
    )
# Singleton instance for global access
config = AppConfig()

