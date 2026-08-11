# B2 configuration findings (live verification, 2026-08-07 → 08)

Found by running the ingestion path against the real Backblaze account and Supabase
project.

## 1. ~~The tenant holding all the data has no bucket configured~~ — APPLIED

`clients_registry.b2_bucket` was `NULL` for `collaborative-process`, the tenant that
owns all 272 `video_summaries` rows and 4,557 `transcript_segments`, so
`_get_client_storage()` returned `None`, the legend never built, and ingestion could
never fire. **Set to `TCP-MASTER` on 2026-08-08 after the owner confirmed it.**

```sql
update public.clients_registry
set    b2_bucket = 'TCP-MASTER',
       b2_prefix = ''
where  slug = 'collaborative-process';
```

Verified afterwards with no overrides of any kind: **271 of 272 sources deliver
transcript text, 0 errors, transcript-ordered-first in all 271**, at 0.27s per source.
The one miss is `PAW MSS DEBRIEF - TRANSCRIPT ONLY.mp4` — a title whose trailing words
are the sidecar role token and which has no matching sidecar in the bucket. It
degrades to an empty bundle; nothing to fix in code.

Still `NULL`: `korevin-solutions` (no B2 data yet) and `test` (`source_kind='gdrive'`).

## 2. Bucket name typo on `test-company`

`clients_registry` records `VPStorage-testcompany`; the bucket that actually exists in
B2 is `VPStorage-testcomapny` (transposed `pa`/`ap`). That bucket is empty, so nothing
depends on it today, but the lookup would fail if it were used.

```sql
update public.clients_registry
set    b2_bucket = 'VPStorage-testcomapny'   -- or rename the bucket in B2 instead
where  slug = 'test-company';
```

## 3. `video_summaries.b2_path` is NULL on all 272 rows (non-blocking)

Resolution falls back to `find_file_by_name(source_file)`, which the legend now serves
from cache, so this costs a dict lookup rather than a scan. Backfilling `b2_path` with
the full path the legend already resolves would remove the fallback entirely. Optional.

## 4. ~~Transcript truncation is the real limiter~~ — ADDRESSED

At `source_file_content_max_chars = 60,000`, **99 of 271 transcripts (36.5%) were cut
mid-sentence**. Raised to **200,000** (`orchestrator/config.py`), which clears the
longest transcript in the corpus. Measured end-to-end afterwards: **0 of 271 sources
truncated**, median ingest 47,168 chars (~11.8k tokens), largest 192,594 (~48k tokens)
— comfortably inside grok-4.3's window.

Truncation, when it does fire, now shrinks only the longest part and keeps the summary
whole, so a cut source still carries a complete overview instead of just its opening
minutes.

Override with `SOURCE_FILE_CONTENT_MAX_CHARS` if prompt cost matters more than recall;
every "dive deeper" turn now sends ~12k tokens of transcript on the median source.

## 5. Local-only: Norton intercepts TLS on this machine

Norton's HTTPS scanning presents its own certificate, so Python rejects every outbound
call with `CERTIFICATE_VERIFY_FAILED` (certifi doesn't help; Node works because
`NODE_EXTRA_CA_CERTS` is already set to `C:\ProgramData\Norton\Antivirus\wscert.pem`).
Affects local runs only — Render is unaffected, and no code change is warranted. To run
locally, point Python at a bundle combining certifi with Norton's CA:

```
SSL_CERT_FILE=<certifi cacert.pem + wscert.pem concatenated>
```
