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
    b2_workspace_bucket_prefix: str = "vpstorage"

    # Server-to-server transcription access. Browser clients must never receive
    # B2 credentials, so B2-link ingestion is proxied through this service.
    transcription_api_url: str = ""
    transcription_api_key: str = ""

    # Self-service storage provisioning. The webhook is an n8n workflow that
    # uses the existing Google Drive OAuth credential to create the intake and
    # completed folders beneath this TCP Shared Drive parent.
    storage_provisioning_webhook_url: str = ""
    storage_provisioning_webhook_secret: str = ""
    google_drive_provisioning_parent_id: str = "1RJQXpLXOE3cXclgpZfvSPt0U648pYuIb"

    # Render-side Google Drive access used to trash transcript/summary exports.
    # Store the complete service-account key JSON as a secret environment value.
    google_drive_service_account_json: str = ""
    # Optional Workspace user for domain-wide delegation. Leave blank when the
    # service account is a member of the Shared Drive itself.
    google_drive_delegated_user: str = ""

    opusclip_api_key: str = ""
    opusclip_api_url: str = "https://api.opus.pro/api"
    # TTL for the signed B2 download URL handed to OpusClip so it can fetch the
    # source video directly. Must outlast OpusClip's download/queue window.
    # Max 604800 (B2's 7-day cap). Default 24h.
    b2_download_url_ttl_seconds: int = 86400
    # "Legend" = local JSON index of a tenant's B2 bucket (normalized basename →
    # full path), so resolving a file is a dict lookup instead of paginating the
    # bucket on every request. See orchestrator/tools/local/b2_legend.py.
    b2_legend_cache_dir: str = ".b2_legend"
    b2_legend_ttl_seconds: int = 900  # rebuild the index when older than 15 min
    # Floor between miss-triggered rebuilds: without it, asking for a file that
    # does not exist would re-sweep the whole bucket every turn.
    b2_legend_min_refresh_seconds: int = 60
    # Fan-out width when fetching a source file + its transcript sidecars.
    # B2's default account limit is 500 req/s, so this is deliberately modest.
    b2_fetch_concurrency: int = 6

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
    # Cap on the transcript text handed to analyze/source_answer. A "dive deeper"
    # turn can pull a source file plus its sidecars; without a cap one long
    # transcript blows the prompt budget.
    # 200k chars ~= 50k tokens, comfortably inside grok-4.3's window and enough
    # for the longest transcript in the corpus (184k). At 60k the cap silently
    # cut 36% of transcripts mid-sentence, which is the opposite of what a
    # "dive deeper" turn is for. Lower it if prompt cost matters more than recall.
    source_file_content_max_chars: int = 200_000
    # Target clip length(s) Opus aims for when curating a full-file project (seconds).
    opus_default_clip_seconds: int = 60
    # Comma-separated list of allowed CORS origins. Use "*" for dev/testing.
    # Set to your frontend URL(s) in production, e.g. "https://app.example.com"
    cors_origins: str = "*"


settings = Settings()  # type: ignore[call-arg]
