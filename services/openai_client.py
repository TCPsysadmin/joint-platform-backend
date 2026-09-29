"""OpenAI transcription client"""
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _seg_field(seg, name: str, default=None):
    """Read a segment field whether the SDK returned a dict or an object."""
    if isinstance(seg, dict):
        return seg.get(name, default)
    return getattr(seg, name, default)


def format_segments(segments: list[dict], time_offset: float = 0.0, fallback_text: str = "") -> str:
    """Render segments as "[HH:MM:SS] text" lines, shifting timestamps by time_offset.
    When there are no segments, fallback_text is prefixed with the chunk start time.
    """
    if segments:
        lines = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if text:
                lines.append(f"[{_fmt_ts(seg.get('start', 0.0) + time_offset)}] {text}")
        return "\n".join(lines)
    # Fallback: no segments returned, prefix with chunk start time
    text = (fallback_text or "").strip()
    return f"[{_fmt_ts(time_offset)}] {text}" if text else ""


class OpenAITranscriber:
    """Client for OpenAI transcription API"""

    def __init__(self, api_key: str):
        # Increase timeout for large audio files (default is 15 minutes)
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=900.0  # 15 minutes timeout
        )

    async def transcribe_segments(self, audio_file_path: str) -> tuple[list[dict], str]:
        """Transcribe an audio file using OpenAI Whisper.
        Returns (segments, text). Each segment is a dict with start/end (seconds from the
        start of this file), text, and the no_speech_prob/avg_logprob confidence fields.
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )

            segments = []
            for seg in getattr(response, "segments", None) or []:
                segments.append({
                    "start": float(_seg_field(seg, "start", 0.0) or 0.0),
                    "end": float(_seg_field(seg, "end", 0.0) or 0.0),
                    "text": _seg_field(seg, "text", "") or "",
                    "no_speech_prob": _seg_field(seg, "no_speech_prob"),
                    "avg_logprob": _seg_field(seg, "avg_logprob"),
                })
            text = getattr(response, "text", None) or ""

            logger.info(f"Transcribed {audio_file_path}")
            return segments, text

        except Exception as e:
            logger.error(f"Transcription failed for {audio_file_path}: {e}")
            raise

    async def transcribe(self, audio_file_path: str, time_offset: float = 0.0) -> str:
        """Transcribe an audio file using OpenAI Whisper, returning timestamped segments.
        time_offset shifts all segment timestamps to account for this chunk's position in the full file.
        """
        segments, text = await self.transcribe_segments(audio_file_path)
        return format_segments(segments, time_offset, text)
