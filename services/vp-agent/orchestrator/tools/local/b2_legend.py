"""Local per-tenant index ("legend") of the files in a client's B2 bucket.

Backblaze's `prefix` is a literal-string match — there is no server-side search
by basename — so resolving "the file called X" used to mean paginating up to
10 pages x 1000 entries on *every* lookup (`B2FileTool.find_file_by_name`).
That is far too slow for an interactive chat turn.

The legend turns that into one full listing sweep, cached on local disk as JSON
and in memory per process, after which a lookup is a dict access.

On-disk format (deliberately dead simple, no schema migrations)::

    {
      "built_at": "2026-08-07T12:00:00+00:00",
      "bucket": "TCP-MASTER",
      "prefix": "",
      "files": {
        "<normalized basename>": [
          {"file_name": "<full path in bucket>", "file_id": ..., "content_length": ...,
           "content_type": ..., "upload_timestamp": ...},
          ...
        ]
      }
    }

The value is a *list* because the normalization key deliberately drops the
extension (so "clip", "clip.mp4" and "clip.txt" all collide) and because the
same basename can legitimately live under two folders. Collisions are resolved
at lookup time, not at build time.

A second, in-memory-only index (`by_base`) keys entries by that name minus a
trailing role token — "transcript", "summary". Real buckets do not name a
transcript after its video with a different extension; they name it
"<video>_transcript.txt". Keying on the extension-stripped basename alone files
a video and its own transcript apart, and they never pair: against the live
bucket that produced 0 sidecar matches across 236 resolved files. The token is
anchored at the end so a title that merely contains the word — "PAW MSS DEBRIEF
- TRANSCRIPT ONLY" — keeps it.

This module owns the filename primitives (extension sets, normalization,
text-file detection) that `b2_file` also uses, so the legend's keys and the
fallback scan can never drift apart.
"""

from __future__ import annotations

import contextlib
import json
import os
import posixpath
import re
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog

from orchestrator.tools.protocols import B2FileEntry

logger = structlog.get_logger(__name__)

# Files we are willing to read as text at all.
TEXT_FILE_EXTENSIONS = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".srt",
    ".text",
    ".txt",
    ".vtt",
}
TEXT_CONTENT_TYPES = {
    "application/json",
    "application/srt",
    "application/vtt",
    "text/csv",
    "text/markdown",
    "text/plain",
    "text/srt",
    "text/vtt",
}
# The subset worth auto-ingesting as "the transcript sitting next to this video".
# .csv/.json/.log are text but are data dumps, not something to paste into a prompt.
TEXT_SIDECAR_EXTENSIONS = {".md", ".srt", ".text", ".txt", ".vtt"}

# Media we must never pull down looking for text: content-type is only known
# after the body has already arrived, and these run to hundreds of megabytes.
MEDIA_FILE_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".m4v",
    ".mov",
    ".mp3",
    ".mp4",
    ".wav",
}
IMAGE_FILE_EXTENSIONS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}

KNOWN_SOURCE_FILE_EXTENSIONS = TEXT_FILE_EXTENSIONS | MEDIA_FILE_EXTENSIONS


def is_text_file(path: str, content_type: str | None) -> bool:
    normalized_type = (content_type or "").split(";", 1)[0].lower()
    if normalized_type.startswith("text/") or normalized_type in TEXT_CONTENT_TYPES:
        return True
    return posixpath.splitext(path.lower())[1] in TEXT_FILE_EXTENSIONS


def is_media_file(path: str) -> bool:
    """True for a path we can rule out as text from its name alone."""
    return posixpath.splitext(path.lower())[1] in MEDIA_FILE_EXTENSIONS


def is_image_file(path: str) -> bool:
    """True for an image extension allowed by the media-library thumbnail signer."""
    return posixpath.splitext(path.lower())[1] in IMAGE_FILE_EXTENSIONS


def is_text_sidecar(file_name: str) -> bool:
    return posixpath.splitext(file_name.lower())[1] in TEXT_SIDECAR_EXTENSIONS


def without_known_extension(value: str) -> str:
    root, extension = posixpath.splitext(value)
    if extension.lower() not in KNOWN_SOURCE_FILE_EXTENSIONS:
        return value
    return root


def normalize_file_match_text(value: str) -> str:
    """Basename, minus a known media/text extension, lowercased, punctuation → spaces."""
    stem = without_known_extension(posixpath.basename(value.strip().strip("/"))).lower()
    return re.sub(r"[^a-z0-9]+", " ", stem).strip()


# Transcripts are not named after their video with a different extension — they
# carry a role suffix ("<video>_transcript.txt", "<video>_summary.txt"). Matching
# on the extension-stripped basename alone therefore files a video and its own
# transcript under two different keys and they never pair. Anchored at the end so
# a title that merely *contains* the word ("PAW MSS DEBRIEF - TRANSCRIPT ONLY")
# keeps it.
_SIDECAR_ROLE_RE = re.compile(r"\s*(transcript|summaryplaud|summary|captions?|subtitles?)$")

# Preference when several sidecars share a stem: a transcript is the content we
# actually want ingested; a summary is a fallback. Lower sorts first.
_SIDECAR_ROLE_RANK = {"transcript": 0, "summary": 1}


def base_match_key(value: str) -> str:
    """`normalize_file_match_text` with a trailing sidecar role token removed.

    This is the key that pairs "X.mp4", "X_transcript.txt" and "X_summary.txt".
    """
    return _SIDECAR_ROLE_RE.sub("", normalize_file_match_text(value)).strip()


def sidecar_role(file_name: str) -> str | None:
    """ "transcript" / "summary" / ... for a sidecar, None if it carries no role."""
    match = _SIDECAR_ROLE_RE.search(normalize_file_match_text(file_name))
    if match is None:
        return None
    role = match.group(1)
    return "summary" if role == "summaryplaud" else role


def sidecar_sort_rank(file_name: str) -> tuple[int, str]:
    """Transcripts before summaries before anything else, then by path.

    Ingestion truncates at a character cap, so a summary sorted ahead of its
    transcript would eat the budget the transcript needs. Ordering has to be a
    property of the *filename*, not of which file happened to resolve first —
    when a video was never uploaded, the source resolves to whichever sidecar
    the index yields, and that can be the summary.
    """
    return _SIDECAR_ROLE_RANK.get(sidecar_role(file_name) or "", 2), file_name


def _sidecar_sort_key(entry: B2FileEntry) -> tuple[int, str]:
    return sidecar_sort_rank(entry.file_name)


@dataclass
class B2Legend:
    """Normalized-basename → entries index for one bucket+prefix."""

    bucket: str
    prefix: str
    built_at: datetime
    files: dict[str, list[B2FileEntry]] = field(default_factory=dict)
    # Derived from `files`, never persisted: role-suffix-stripped key → entries.
    _by_base: dict[str, list[B2FileEntry]] | None = field(default=None, repr=False, compare=False)

    @property
    def by_base(self) -> dict[str, list[B2FileEntry]]:
        """Index that pairs a media file with its `_transcript` / `_summary` twins."""
        if self._by_base is None:
            grouped: dict[str, list[B2FileEntry]] = {}
            for entries in self.files.values():
                for entry in entries:
                    grouped.setdefault(base_match_key(entry.file_name), []).append(entry)
            self._by_base = grouped
        return self._by_base

    # ── freshness ────────────────────────────────────────────────────────────

    def age_seconds(self, *, now: datetime | None = None) -> float:
        return ((now or datetime.now(UTC)) - self.built_at).total_seconds()

    def is_stale(self, ttl_seconds: int, *, now: datetime | None = None) -> bool:
        return self.age_seconds(now=now) >= ttl_seconds

    def matches(self, bucket: str, prefix: str) -> bool:
        return self.bucket == bucket and self.prefix == prefix

    @property
    def entry_count(self) -> int:
        return sum(len(v) for v in self.files.values())

    # ── lookup ───────────────────────────────────────────────────────────────

    def lookup(self, file_name: str, *, text_only: bool = False) -> B2FileEntry | None:
        """Resolve a full path or basename to a single entry.

        Preference order: exact full-path match, then exact basename match, then
        the most recently uploaded candidate. `text_only` restricts the result to
        text sidecars (a transcript, not the video it sits next to).
        """
        target = file_name.strip().strip("/")
        if not target:
            return None
        candidates = self.files.get(normalize_file_match_text(target)) or []
        if not candidates:
            # Nothing under the literal name: widen to the role-stripped key, so
            # asking for "X.mp4" can still find "X_transcript.txt" when the video
            # itself was never uploaded.
            candidates = self.by_base.get(base_match_key(target)) or []
        if text_only:
            candidates = [e for e in candidates if is_text_sidecar(e.file_name)]
        if not candidates:
            return None

        if text_only:
            return sorted(candidates, key=_sidecar_sort_key)[0]

        target_basename = posixpath.basename(target)
        for entry in candidates:
            if entry.file_name == target:
                return entry
        for entry in candidates:
            if posixpath.basename(entry.file_name) == target_basename:
                return entry
        return max(candidates, key=lambda e: (e.upload_timestamp or 0, e.file_name))

    def text_siblings(self, file_name: str) -> list[B2FileEntry]:
        """Text sidecars belonging to this file (its transcript/summary twins).

        Keyed on the role-stripped stem, so "X.mp4" finds "X_transcript.txt".
        Excludes the file itself; transcripts sort ahead of summaries so the
        ingestion character cap is spent on the transcript first.
        """
        target = file_name.strip().strip("/")
        candidates = self.by_base.get(base_match_key(target)) or []
        return sorted(
            (e for e in candidates if e.file_name != target and is_text_sidecar(e.file_name)),
            key=_sidecar_sort_key,
        )

    # ── (de)serialization ────────────────────────────────────────────────────

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "built_at": self.built_at.isoformat(),
            "bucket": self.bucket,
            "prefix": self.prefix,
            "files": {
                key: [
                    {
                        "file_name": e.file_name,
                        "file_id": e.file_id,
                        "content_length": e.content_length,
                        "content_type": e.content_type,
                        "upload_timestamp": e.upload_timestamp,
                    }
                    for e in entries
                ]
                for key, entries in self.files.items()
            },
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> B2Legend:
        files: dict[str, list[B2FileEntry]] = {}
        for key, rows in (data.get("files") or {}).items():
            entries = [
                B2FileEntry(
                    file_name=str(row["file_name"]),
                    file_id=str(row["file_id"]) if row.get("file_id") else None,
                    action="upload",
                    content_length=int(row.get("content_length") or 0),
                    content_type=str(row["content_type"]) if row.get("content_type") else None,
                    upload_timestamp=(
                        int(row["upload_timestamp"])
                        if row.get("upload_timestamp") is not None
                        else None
                    ),
                )
                for row in rows
                if isinstance(row, dict) and row.get("file_name")
            ]
            if entries:
                files[str(key)] = entries
        return cls(
            bucket=str(data.get("bucket") or ""),
            prefix=str(data.get("prefix") or ""),
            built_at=datetime.fromisoformat(str(data["built_at"])),
            files=files,
        )

    @classmethod
    def from_entries(
        cls,
        entries: list[B2FileEntry],
        *,
        bucket: str,
        prefix: str,
        built_at: datetime | None = None,
    ) -> B2Legend:
        files: dict[str, list[B2FileEntry]] = {}
        for entry in entries:
            # "folder" placeholders, hide markers and in-progress large uploads
            # are not real files — indexing them would resolve to dead paths.
            if entry.action != "upload" or not entry.file_name:
                continue
            key = normalize_file_match_text(entry.file_name)
            if not key:
                continue
            files.setdefault(key, []).append(entry)
        return cls(
            bucket=bucket,
            prefix=prefix,
            built_at=built_at or datetime.now(UTC),
            files=files,
        )


# ── disk persistence ─────────────────────────────────────────────────────────


def legend_path(cache_dir: str | Path, client_id: UUID | str) -> Path:
    return Path(cache_dir) / f"{client_id}.json"


def load_legend(path: Path) -> B2Legend | None:
    """Read a persisted legend, treating anything unreadable as "no legend".

    A truncated or hand-edited cache file must degrade to a rebuild, never take
    a chat turn down.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("b2_legend_unreadable", path=str(path), error=str(exc))
        return None
    try:
        return B2Legend.from_json_dict(data)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("b2_legend_corrupt", path=str(path), error=str(exc))
        return None


def save_legend(path: Path, legend: B2Legend) -> bool:
    """Persist atomically (temp file in the same dir + os.replace).

    Several sessions for one tenant can rebuild concurrently; a plain write
    would eventually leave a half-written file that every later lookup trips on.
    Returns False instead of raising — a cache we can't write is not a reason to
    fail the turn.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(legend.to_json_dict(), handle)
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
    except OSError as exc:
        logger.warning("b2_legend_save_failed", path=str(path), error=str(exc))
        return False
    return True
