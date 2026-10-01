-- LangChain PostgreSQL Setup for Supabase
-- Run this in your Supabase SQL Editor

-- =============================================================================
-- VECTOR SEARCH SETUP (tcpdb_v2)
-- =============================================================================

-- 1) Ensure pgvector is available
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) New table (no spaces in name)
CREATE TABLE IF NOT EXISTS tcp_db_v2 (
  id BIGSERIAL PRIMARY KEY,
  content TEXT,
  metadata JSONB,
  embedding VECTOR(1536)  -- OpenAI text-embedding-3-small
);

-- 3) Indexes
-- 3a) HNSW (cosine) for fast ANN search
CREATE INDEX IF NOT EXISTS tcp_db_v2_embedding_hnsw
ON tcp_db_v2
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 3b) GIN on metadata for quick @> filters
CREATE INDEX IF NOT EXISTS tcp_db_v2_metadata_gin
ON tcp_db_v2
USING gin (metadata jsonb_path_ops);

-- 4) Simple search RPC (uses current session ef_search)
CREATE OR REPLACE FUNCTION tcpdb_v2_search (
  query_embedding VECTOR(1536),
  match_count INT DEFAULT 5,
  filter JSONB DEFAULT '{}'
) RETURNS TABLE (
  id BIGINT,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE SQL
AS $$
  SELECT
    id,
    content,
    metadata,
    1 - (embedding <=> query_embedding) AS similarity
  FROM tcp_db_v2
  WHERE metadata @> filter
  ORDER BY embedding <=> query_embedding
  LIMIT match_count
$$;

-- 5) Tuned search RPC (lets you set ef_search per call)
CREATE OR REPLACE FUNCTION tcpdb_v2_search_tuned (
  query_embedding VECTOR(1536),
  match_count INT DEFAULT 5,
  filter JSONB DEFAULT '{}',
  ef_search INT DEFAULT 60   -- try 40–100
) RETURNS TABLE (
  id BIGINT,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  PERFORM set_config('hnsw.ef_search', ef_search::text, true);
  RETURN QUERY
    SELECT
      id,
      content,
      metadata,
      1 - (tcp_db_v2.embedding <=> query_embedding) AS similarity
    FROM tcp_db_v2
    WHERE metadata @> filter
    ORDER BY tcp_db_v2.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 6) Refresh stats after bulk ingest
ANALYZE tcp_db_v2;

-- =============================================================================
-- CHAT MEMORY SETUP (chat_memory_v2)
-- =============================================================================

-- 7) Create optimized chat memory table
CREATE TABLE IF NOT EXISTS chat_memory_v2 (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    message_type VARCHAR(20) NOT NULL CHECK (message_type IN ('human', 'ai')),
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 8) Optimized indexes for chat memory
-- Composite index for fast session + time queries
CREATE INDEX IF NOT EXISTS idx_chat_memory_v2_session_created 
ON chat_memory_v2(session_id, created_at);

-- Additional index for cleanup operations
CREATE INDEX IF NOT EXISTS idx_chat_memory_v2_created_at 
ON chat_memory_v2(created_at);

-- 9) Enable Row Level Security (RLS)
ALTER TABLE chat_memory_v2 ENABLE ROW LEVEL SECURITY;

-- 10) Create policy for public access
CREATE POLICY "Allow public access to chat_memory_v2" ON chat_memory_v2
    FOR ALL USING (true);

-- =============================================================================
-- CHAT MEMORY FUNCTIONS
-- =============================================================================

-- 11) Get conversation history (optimized)
CREATE OR REPLACE FUNCTION get_conversation_history(
  session_id_param VARCHAR,
  message_limit INT DEFAULT 10
) RETURNS TABLE (
  message_type VARCHAR,
  content TEXT,
  created_at TIMESTAMP WITH TIME ZONE
) 
LANGUAGE SQL
AS $$
  SELECT 
    message_type,
    content,
    created_at
  FROM chat_memory_v2
  WHERE session_id = session_id_param
  ORDER BY created_at ASC
  LIMIT message_limit;
$$;

-- 12) Add user message
CREATE OR REPLACE FUNCTION add_user_message(
  session_id_param VARCHAR,
  message_content TEXT
) RETURNS VOID
LANGUAGE SQL
AS $$
  INSERT INTO chat_memory_v2 (session_id, message_type, content, created_at)
  VALUES (session_id_param, 'human', message_content, NOW());
$$;

-- 13) Add AI message
CREATE OR REPLACE FUNCTION add_ai_message(
  session_id_param VARCHAR,
  message_content TEXT
) RETURNS VOID
LANGUAGE SQL
AS $$
  INSERT INTO chat_memory_v2 (session_id, message_type, content, created_at)
  VALUES (session_id_param, 'ai', message_content, NOW());
$$;

-- 14) Clear conversation history
CREATE OR REPLACE FUNCTION clear_conversation_history(
  session_id_param VARCHAR
) RETURNS VOID
LANGUAGE SQL
AS $$
  DELETE FROM chat_memory_v2 
  WHERE session_id = session_id_param;
$$;

-- 15) Clean old messages (older than 30 days)
CREATE OR REPLACE FUNCTION clean_old_chat_memory()
RETURNS VOID AS $$
BEGIN
    DELETE FROM chat_memory_v2 
    WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- LEGACY SUPPORT (Keep old table for migration)
-- =============================================================================

-- 16) Keep the old table for backward compatibility
CREATE TABLE IF NOT EXISTS langchain_chat_history (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    message_type VARCHAR(20) NOT NULL CHECK (message_type IN ('human', 'ai')),
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 17) Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_langchain_session_id ON langchain_chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_langchain_created_at ON langchain_chat_history(created_at);

-- 18) Enable Row Level Security (RLS)
ALTER TABLE langchain_chat_history ENABLE ROW LEVEL SECURITY;

-- 19) Create policy for public access
CREATE POLICY "Allow public access to langchain_chat_history" ON langchain_chat_history
    FOR ALL USING (true);

-- 20) Optional: Create a function to clean old messages (older than 30 days)
CREATE OR REPLACE FUNCTION clean_old_langchain_history()
RETURNS VOID AS $$
BEGIN
    DELETE FROM langchain_chat_history 
    WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;
