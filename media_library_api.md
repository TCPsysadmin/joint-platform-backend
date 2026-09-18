# Media library API

The media library exposes each source video as a folder-like item. A frontend can
render `thumbnail.url` as the folder cover and open the item to show its video,
summary, and transcript.

All requests require the normal Supabase bearer token. Results are automatically
limited to the authenticated user's tenant.

## List folders

`GET /media?limit=50&offset=0`

```json
{
  "items": [
    {
      "id": "video-123",
      "kind": "video_folder",
      "name": "Founder interview",
      "thumbnail": {
        "url": "https://cdn.example.com/video-123.jpg",
        "b2_path": "thumbnails/video-123.jpg"
      },
      "video": {
        "source_video_id": "video-123",
        "source_file": "founder-interview.mp4",
        "b2_path": "videos/founder-interview.mp4",
        "duration_seconds": 1800,
        "recorded_at": "2026-07-01",
        "has_timestamps": true
      },
      "summary": {
        "text": "An interview about...",
        "topics": ["founders", "growth"],
        "speakers": ["Alex"],
        "quality_score": 0.91
      },
      "created_at": "2026-07-01T12:00:00Z",
      "updated_at": "2026-07-01T12:00:00Z"
    }
  ],
  "count": 1,
  "limit": 50,
  "offset": 0
}
```

## Open a folder

`GET /media/{source_video_id}`

Returns the same folder fields plus:

```json
{
  "transcript": {
    "segment_count": 2,
    "segments": [
      {
        "segment_id": "...",
        "chunk_index": 0,
        "start_seconds": 0,
        "end_seconds": 45,
        "speaker": "Alex",
        "transcript_text": "...",
        "word_count": 120
      }
    ],
    "text": "The complete transcript joined in segment order..."
  }
}
```

## Thumbnail ingestion

`video_summaries.thumbnail_url` supports a public/CDN URL.
`video_summaries.thumbnail_b2_path` supports a private thumbnail stored alongside
the media. When only the private path exists, `GET /media` and
`GET /media/{source_video_id}` fill the response's `thumbnail.url` with a
short-lived, tenant-scoped B2 URL. Existing records may leave both fields null;
the frontend should show a video placeholder until ingestion generates a thumbnail.

The visual folder grid belongs in the frontend application. This repository only
provides the authenticated data contract.

## Download the source video

`GET /media/{source_video_id}/video-url`

Returns a short-lived, tenant-scoped B2 download URL:

```json
{
  "url": "https://f000.backblazeb2.com/file/...",
  "filename": "founder-interview.mp4",
  "expires_in": 86400
}
```

## Delete videos

`POST /media/bulk-delete`

Workspace owners and admins can permanently delete up to 100 videos at once:

```json
{
  "source_video_ids": ["video-123", "video-456"]
}
```

The operation is tenant-scoped. It first moves the corresponding transcript and
summary exports to Google Drive Trash, removes the source videos and private
thumbnails from the workspace's B2 storage, then atomically deletes their
ingestion manifests, transcript segments, and summary rows.

If Drive or B2 cleanup fails, the endpoint returns `502` and leaves the database
records in place so the deletion can be retried. Drive cleanup is performed
directly by this backend through Google Drive API v3; n8n is not involved.
Configure `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON` in Render with the complete
service-account key JSON, and grant that account access to the Shared Drive.
If the Workspace uses domain-wide delegation instead, also set
`GOOGLE_DRIVE_DELEGATED_USER` to the delegated Workspace user email.

A successful response is:

```json
{
  "deleted": 2,
  "source_video_ids": ["video-123", "video-456"],
  "warnings": []
}
```

Run `media_management_migration.sql` in Supabase before deploying this endpoint.
