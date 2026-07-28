from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+psycopg2://weft_user:weft_123@localhost:5432/weft_db"
    
    # Models
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    
    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    
    # Data Paths
    RAW_DATA_ZIP: str = "Data/Raw Data/reddyranbeer openAI Data.zip"
    EXTRACTED_DATA_DIR: str = "Data/Extracted Data/"
    MERGED_CONVERSATIONS_FILE: str = "conversations.json"
    VAULT_DIR: str = "vault/conversations"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
