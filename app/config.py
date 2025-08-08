# app/config.py
from pydantic_settings import BaseSettings


# This is the original code. We are commenting it out for the test.
class Settings(BaseSettings):
    # API Keys and Tokens
    GOOGLE_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    BEARER_TOKEN: str

    # Model and Index Configuration
    PINECONE_INDEX_NAME: str = "hackrx-index"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    # GENERATIVE_MODEL: str = "gemini-2.0-flash"
    GENERATIVE_MODEL: str = "gemini-2.0-flash"
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8" 
# Create a single instance of the settings to be used across the app
settings = Settings()