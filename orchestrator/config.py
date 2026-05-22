from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    supabase_db_url: str
    openai_api_key: str

    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""

    b2_key_id: str = ""
    b2_application_key: str = ""

    opusclip_api_key: str = ""
    opusclip_api_url: str = "https://api.opus.pro/api"

    max_critique_iterations: int = 3
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"
    log_level: str = "INFO"
    require_brand_doctrine: bool = True
    context_window_messages: int = 12  # sliding window: last N messages sent to LLM
    retrieve_match_count: int = 15  # hybrid_search_transcripts top-K (ranked by relevance)
    # Comma-separated list of allowed CORS origins. Use "*" for dev/testing.
    # Set to your frontend URL(s) in production, e.g. "https://app.example.com"
    cors_origins: str = "*"


settings = Settings()  # type: ignore[call-arg]
