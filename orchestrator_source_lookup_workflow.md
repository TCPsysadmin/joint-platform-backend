# Orchestrator Source Lookup Workflow

This workflow handles prompts like:

`The Video is TCP001_DITL_20250502 - JUST TRY what is it about`

or:

`What is this about https://f005.backblazeb2.com/file/TCP-MASTER/TCP001_DITL/TCP001_DITL_20250502%20-%20JUST%20TRY.mp3`

or:

`Give me interesting parts from TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr. Kieschnick`

## 1. Chat Request

`POST /chat` receives the user message and loads any uploaded session documents.

It then starts the LangGraph workflow with:

- `messages`
- `session_id`
- `session_documents`

## 2. Intent Routing

`route_intent` still asks the LLM to classify the message, but it now also runs deterministic source detection over the latest user message.

It extracts source references from:

- Backblaze/public URLs
- File names such as `.mp3`, `.mp4`, `.txt`, `.vtt`
- Phrases like `The video is <title>`
- Coded titles like `TCP001_DITL_20250502 - JUST TRY`

If the LLM incorrectly returns `chitchat` but the message contains a source reference, the router overrides it to `new_request`.

For “what is it about” style prompts, it sets:

- `source_reference`: the exact title, filename, or URL
- `source_task`: `source_answer`

For clip requests from a source, it sets:

- `source_task`: `clip_recommendation`

## 3. Source Resolution

`retrieve` sees `source_reference` and does not run a global transcript search.

Instead it calls:

`runtime.search_tool.resolve_source_video(source_reference)`

The Supabase search tool checks:

- `video_summaries.source_video_id`
- `video_summaries.title`
- `video_summaries.source_file`
- `video_summaries.b2_path`
- `transcript_segments.source_video_id`
- `transcript_segments.source_title`
- `transcript_segments.source_file`

Backblaze URLs are decoded into useful variants, including object path, basename, and filename stem.

Title-style references also produce a date-code variant. For example:

`TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr. Kieschnick`

also searches:

`TCP003_MEETINGS_20230724`

This keeps lookup resilient when the indexed title has punctuation or speaker-name differences.

If direct Supabase lookup fails, `retrieve` asks `b2_file.py` to locate a matching object in Backblaze B2 by name/stem. When B2 finds a file, the resolver retries Supabase lookup using the real B2 path and basename.

If B2 finds the file but Supabase still has no matching transcript source, the response says the B2 file exists but indexed transcript chunks could not be matched. It does not incorrectly say the file itself was missing.

## 4. Transcript Loading From Vector DB

Once the source resolves to `source_video_id`, `retrieve` calls:

`list_transcript_segments_for_video(source_video_id=...)`

This loads the transcript chunks for that exact source only.

No fallback global search is performed for source-specific requests.

## 5. Optional B2 Text Fetch

`fetch_source` calls:

`runtime.file_tool.fetch_source_file(source_video_id)`

`b2_file.py` resolves the source to the configured Backblaze bucket/path through `video_summaries` and `clients_registry`.

B2 content is only returned when the downloaded file is text-like, such as:

- `.txt`
- `.md`
- `.srt`
- `.vtt`
- `.json`
- `text/*`

Binary media like `.mp3` or `.mp4` is skipped so raw audio/video bytes are not sent to the LLM.

If B2 text is unavailable, the workflow still uses transcript segments from Supabase.

If Supabase source resolution fails but B2 finds a text-like file, `retrieve` can carry that downloaded text forward as a no-timestamp source fallback.

## 6. Source Answer Branch

If `source_task` is `source_answer`, the graph routes:

`fetch_source -> source_answer -> END`

The `source_answer` node builds an answer from:

- B2 text content, when available
- otherwise transcript segments from Supabase

It answers what the source is about without recommending clips.

## 7. Clip Recommendation Branch

If `source_task` is `clip_recommendation`, the graph routes:

`fetch_source -> analyze -> critique -> recommend -> post_stub`

The clip path remains constrained to the resolved `source_video_id`.

## 8. Source Not Found

If the source cannot be resolved, the graph routes:

`retrieve -> source_not_found -> END`

The user receives a clear message that the exact source could not be found. The orchestrator does not silently search unrelated videos.
