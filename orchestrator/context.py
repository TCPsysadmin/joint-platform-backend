from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import BaseMessage

# Nouns the creator uses for something we already showed them.
_CLIP_NOUN = r"(?:clip|option|suggestion|recommendation|pick|cut|moment)"
_ORDINAL = r"(?:first|second|third|fourth|fifth|last|previous|latter|former)"
_SMALL_NUMBER = r"(?:\d+|one|two|three|four|five)"

# Matches messages that point at a clip already on screen — by index ("clip 2",
# "option #3", "number 1"), by ordinal ("the second one"), or by demonstrative
# ("that clip", "this one"). Deliberately narrow: bare "tell me more" or
# "elaborate" must NOT match, because those are answered fine by chat_response
# and routing them through analyze costs three extra LLM calls.
_CLIP_REFERENCE_RE = re.compile(
    r"\b(?:"
    rf"{_CLIP_NOUN}s?\s*#?\s*{_SMALL_NUMBER}"
    rf"|{_ORDINAL}\s+(?:{_CLIP_NOUN}|one)"
    rf"|(?:that|this|those|these)\s+(?:{_CLIP_NOUN}|one)"
    rf"|(?:number|no\.|#)\s*{_SMALL_NUMBER}"
    r")\b",
    re.IGNORECASE,
)


# Signals that the creator wants more/different, rather than wanting us to act on
# what's pending. Paired with a clip reference below.
_REFINEMENT_CUE_RE = re.compile(
    r"\b(?:dive|go|dig)\s+deeper\b"
    r"|\bdeeper\b"
    r"|\btell\s+me\s+more\b"
    r"|\bmore\s+(?:about|detail|details|on|depth)\b"
    r"|\b(?:expand|elaborate|explain|unpack|breakdown|compare)\b"
    r"|\b(?:why|how\s+come|what\s+about)\b"
    r"|\b(?:shorter|longer|punchier|tighter|snappier|instead|different)\b",
    re.IGNORECASE,
)


def build_context_window(messages: list[BaseMessage], window_size: int) -> list[BaseMessage]:
    """Return the last `window_size` messages for LLM prompt construction.

    Applied at node level so the full history is preserved in the checkpointer
    (for session replay / GET /sessions/{id}/messages) while keeping token
    usage bounded on long sessions.

    window_size=0 or window_size >= len(messages) → returns messages unchanged.
    """
    if window_size <= 0 or len(messages) <= window_size:
        return messages
    return messages[-window_size:]


def references_prior_clip(text: str) -> bool:
    """True when the message points at a clip that was already presented.

    Used to rescue a follow-up the intent classifier labelled `new_request`, which
    would otherwise wipe the clips being referred to. A false positive here is cheap
    — it only routes the turn through `analyze` with context intact. Do NOT use this
    alone to gate publishing: "make this clip" and "post that one" match it and are
    approvals. Use `is_clip_refinement_request` there instead.
    """
    return bool(_CLIP_REFERENCE_RE.search(text or ""))


def is_clip_refinement_request(text: str) -> bool:
    """True when the message points at a shown clip *and* asks for more/different.

    Requiring both conjuncts is what separates "yes, let's dive deeper on clip 2"
    (a refinement wearing an approval's clothes) from "yes, make this clip" (a real
    approval). Anything that fails this test still goes to the LLM classifier.
    """
    return references_prior_clip(text) and bool(_REFINEMENT_CUE_RE.search(text or ""))


_ORDINAL_POSITIONS = {
    "first": 1,
    "one": 1,
    "second": 2,
    "two": 2,
    "third": 3,
    "three": 3,
    "fourth": 4,
    "four": 4,
    "fifth": 5,
    "five": 5,
}

# The same shapes `_CLIP_REFERENCE_RE` matches, but capturing which one.
_CLIP_POSITION_RE = re.compile(
    rf"\b(?:{_CLIP_NOUN}s?\s*#?\s*(?P<index>\d+)"
    rf"|(?:number|no\.|#)\s*(?P<index2>\d+)"
    rf"|(?P<word>{_ORDINAL})\s+(?:{_CLIP_NOUN}|one)"
    rf"|{_CLIP_NOUN}s?\s+(?P<word2>{_ORDINAL}))\b",
    re.IGNORECASE,
)


def referenced_clip_position(text: str) -> int | None:
    """The 1-based position of the clip a message points at, if it names one.

    "clip 2" / "option #3" / "the second one" → 2 / 3 / 2. Returns None for
    references that don't identify a position ("that one", "the last one") —
    callers fall back to their own default rather than guessing.
    """
    match = _CLIP_POSITION_RE.search(text or "")
    if match is None:
        return None
    digits = match.group("index") or match.group("index2")
    if digits:
        position = int(digits)
        return position if position > 0 else None
    word = (match.group("word") or match.group("word2") or "").lower()
    return _ORDINAL_POSITIONS.get(word)


def summarize_prior_clips(
    clips: list[dict[str, Any]],
    *,
    max_clips: int = 6,
) -> list[dict[str, Any]]:
    """Compact the previous turn's clip candidates for re-injection into a prompt.

    Keeps only the fields needed to resolve an ordinal reference ("clip 2") back
    to a concrete moment; drops b-roll/asset payloads, which are large and are
    regenerated for whichever clip ends up primary.
    """
    keep = (
        "video_id",
        "segment_id",
        "start_seconds",
        "end_seconds",
        "has_timestamps",
        "hook_quote",
        "rationale",
        "source_title",
        "source_file",
    )
    compact: list[dict[str, Any]] = []
    for index, clip in enumerate(clips[:max_clips], start=1):
        entry: dict[str, Any] = {"position": index}
        entry.update({k: clip[k] for k in keep if k in clip})
        compact.append(entry)
    return compact


def build_prior_context_summary(state: dict[str, Any]) -> str | None:
    """A short, plain-text inventory of what is already in the agent's state.

    Given to the intent classifier so it can tell "start over" from "the second
    one" — without it, the classifier only sees message text and has no idea
    that clips are sitting in context waiting to be referred to.
    """
    lines: list[str] = []

    clips = list(state.get("candidate_clips") or [])
    if clips:
        lines.append(f"{len(clips)} clip option(s) from the previous turn are still in context:")
        for index, clip in enumerate(clips, start=1):
            hook = str(clip.get("hook_quote") or "").strip().replace("\n", " ")
            lines.append(f"  Option {index}: {hook[:140]}" if hook else f"  Option {index}")

    if state.get("final_recommendation"):
        lines.append("A recommendation has already been shown to the user in this session.")

    source_metadata = state.get("source_metadata") or {}
    if isinstance(source_metadata, dict):
        title = source_metadata.get("title") or source_metadata.get("source_file")
        if title:
            lines.append(f"Resolved source video: {title}")

    segments = state.get("retrieved_segments") or []
    if segments:
        lines.append(f"{len(segments)} transcript segment(s) are already retrieved and reusable.")

    documents = state.get("session_documents") or []
    names = [str(doc.get("filename") or "document") for doc in documents if isinstance(doc, dict)]
    if names:
        lines.append(f"Uploaded session documents in context: {', '.join(names[:5])}")

    if not lines:
        return None

    return (
        "# Prior Session Context\n"
        "This is what the agent already has in memory for this session. Use it to "
        "decide whether the latest message refers back to it.\n" + "\n".join(lines)
    )
