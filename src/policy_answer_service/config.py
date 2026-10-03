from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = PROJECT_ROOT / "data"
    policy_docs_dir: Path = PROJECT_ROOT / "data" / "policies"
    golden_set_path: Path = PROJECT_ROOT / "evals" / "golden_set.jsonl"
    embedding_cache_path: Path = PROJECT_ROOT / "data" / "embedding_cache.sqlite"
    
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = 5

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = Field(default="openrouter", description="e.g. 'openrouter'")
    llm_base_url: str = Field(default="https://openrouter.ai/api/v1")
    llm_model: str = "anthropic/claude-sonnet-4.6"
    llm_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")

    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = "INFO"

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — import and call this, don't instantiate Settings() directly."""
    return Settings()