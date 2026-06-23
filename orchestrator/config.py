from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    supabase_db_url: str

    # xAI (Grok) — powers the chat/reasoning LLM. Get a key at https://console.x.ai
    xai_api_key: str
    xai_api_base: str = "https://api.x.ai/v1"

    # OpenAI — used ONLY for embeddings. xAI has no embeddings endpoint, and the
    # transcript vectors in the DB were generated with `embedding_model` below, so
    # retrieval must keep using the same OpenAI model to stay in the same vector space.
    openai_api_key: str

    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""

    b2_key_id: str = ""
    b2_application_key: str = ""

    opusclip_api_key: str = ""
    opusclip_api_url: str = "https://api.opus.pro/api"
    # TTL for the signed B2 download URL handed to OpusClip so it can fetch the
    # source video directly. Must outlast OpusClip's download/queue window.
    # Max 604800 (B2's 7-day cap). Default 24h.
    b2_download_url_ttl_seconds: int = 86400

    max_critique_iterations: int = 3
    embedding_model: str = "text-embedding-3-small"  # OpenAI embeddings (see note above)
    llm_model: str = "grok-4"  # xAI Grok chat model ID
    log_level: str = "INFO"
    require_brand_doctrine: bool = True
    context_window_messages: int = 12  # sliding window: last N messages sent to LLM
    retrieve_match_count: int = 15  # hybrid_search_transcripts top-K (ranked by relevance)
    clip_candidate_count: int = 4  # how many ranked clip candidates analyze surfaces
    session_document_max_upload_bytes: int = 2_000_000
    session_document_context_chars: int = 12_000
    session_document_max_count: int = 5
    # Target clip length(s) Opus aims for when curating a full-file project (seconds).
    opus_default_clip_seconds: int = 60
    # Comma-separated list of allowed CORS origins. Use "*" for dev/testing.
    # Set to your frontend URL(s) in production, e.g. "https://app.example.com"
    cors_origins: str = "*"


settings = Settings()  # type: ignore[call-arg]
