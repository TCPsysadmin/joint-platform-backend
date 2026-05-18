from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str
    supabase_service_key: str
    supabase_db_url: str
    openai_api_key: str

    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""

    max_critique_iterations: int = 3
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"
    use_mcp_tools: bool = False
    log_level: str = "INFO"
    require_brand_doctrine: bool = True


settings = Settings()  # type: ignore[call-arg]
