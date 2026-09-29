"""Regression tests for transcripts missing their final seconds/minutes.

No OpenAI calls are made: the transcription API is mocked. Fixtures are generated
with ffmpeg into a temp dir, so ffmpeg/ffprobe must be on PATH (tests skip otherwise).

Run with either:
    python -m pytest tests
    python -m unittest discover -s tests
"""
import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The worker clears its temp dir on construction and reads TEMP_STORAGE_DIR at import,
# so point it at a private dir before importing anything from services.
_TEMP_ROOT = tempfile.mkdtemp(prefix="transcript_tail_tests_")
os.environ["TEMP_STORAGE_DIR"] = _TEMP_ROOT


def _stub_missing_modules():
    """Allow the tests to run without the full production dependency set installed.
    Only modules that fail to import are stubbed; none of them are exercised here.
    """
    class _Placeholder:
        def __init__(self, *args, **kwargs):
            pass

    def ensure(name, attrs, base=_Placeholder):
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            for attr in attrs:
                setattr(module, attr, type(attr, (base,), {}))
            sys.modules[name] = module

    ensure("pydub", ["AudioSegment"])
    ensure("openai", ["AsyncOpenAI"])
    ensure("httpx", ["AsyncClient"])
    try:
        import b2sdk.v2  # noqa: F401
        import b2sdk.exception  # noqa: F401
    except ImportError:
        pkg = types.ModuleType("b2sdk")
        pkg.__path__ = []
        sys.modules["b2sdk"] = pkg
        ensure("b2sdk.v2", ["B2Api", "InMemoryAccountInfo"])
        ensure("b2sdk.exception", ["NonExistentBucket", "FileNotPresent", "B2Error"], Exception)


_stub_missing_modules()

from services import media_processor as mp_module  # noqa: E402
from services import transcription_worker as tw_module  # noqa: E402
from services.media_processor import MediaProcessor, plan_chunk_count  # noqa: E402
from services.upload_manager import UploadError, UploadManager  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def ffmpeg(*args):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
    )


def probe_duration(path):
    """What ffprobe reports from the container header (may be a bitrate estimate)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def decoded_duration(path):
    """Ground truth: decode every frame and read the final timestamp."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path, "-map", "0:a:0", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", out.stderr)
    h, m, s = times[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def make_tone(path, seconds):
    ffmpeg("-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
           "-ac", "1", "-c:a", "libmp3lame", "-q:a", "2", path)


def make_vbr_without_xing(path, loud_seconds=30, quiet_seconds=70):
    """VBR mp3 with no Xing header: loud noise first (high bitrate), then silence
    (low bitrate). ffprobe extrapolates duration from the early bitrate and comes up short.
    """
    ffmpeg(
        "-f", "lavfi", "-i", f"anoisesrc=d={loud_seconds}:c=white:a=0.8",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-filter_complex", f"[1]atrim=0:{quiet_seconds}[s];[0][s]concat=n=2:v=0:a=1",
        "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-q:a", "0", "-write_xing", "0",
        path,
    )


class FakeJobManager:
    def __init__(self):
        self.updates = []

    def update_job(self, job_id, **kwargs):
        self.updates.append(kwargs)

    def update_progress(self, job_id, completed, total):
        pass


def make_worker():
    worker = tw_module.TranscriptionWorker(FakeJobManager(), openai_api_key="test-key")
    return worker


class FakeTranscriptions:
    """Stands in for client.audio.transcriptions; replies come from a callback."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def create(self, model, file, response_format):
        self.calls.append({"model": model, "path": file.name, "format": response_format})
        return self.reply(file.name, len(self.calls))


def install_fake_api(worker, reply):
    transcriptions = FakeTranscriptions(reply)
    worker.openai_client.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions))
    return transcriptions


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        if not HAVE_FFMPEG:
            self.skipTest("ffmpeg/ffprobe not on PATH")
        self.tmp = tempfile.mkdtemp(dir=_TEMP_ROOT)
        self._saved_chunk = tw_module.CHUNK_DURATION_SECONDS

    def tearDown(self):
        tw_module.CHUNK_DURATION_SECONDS = self._saved_chunk
        shutil.rmtree(self.tmp, ignore_errors=True)


class ChunkPlanningTests(unittest.TestCase):
    def test_ceil_and_no_extra_chunk_at_exact_multiple(self):
        self.assertEqual(plan_chunk_count(59, 600), 1)
        self.assertEqual(plan_chunk_count(600, 600), 1)
        self.assertEqual(plan_chunk_count(1200, 600), 2)  # int(d/c)+1 gave 3
        self.assertEqual(plan_chunk_count(1800, 600), 3)
        self.assertEqual(plan_chunk_count(1201, 600), 3)
        self.assertEqual(plan_chunk_count(1799.9, 600), 3)

    def test_sub_second_remainder_is_folded_into_last_chunk(self):
        # mp3 padding makes an exact multiple probe a few ms long
        self.assertEqual(plan_chunk_count(1200.03, 600), 2)
        self.assertEqual(plan_chunk_count(1200.6, 600), 3)


class ChunkCreationTests(TempDirTestCase):
    def test_chunks_cover_whole_file_at_exact_multiple(self):
        src = os.path.join(self.tmp, "tone.mp3")
        make_tone(src, 30)
        mp = MediaProcessor()
        duration, num_chunks = asyncio.run(mp.get_chunk_info(src, chunk_duration=10))
        # ~30.0s probes a few ms long; int(d/c)+1 would plan an empty 4th chunk
        self.assertEqual(num_chunks, 3)

        total = 0.0
        for i in range(num_chunks):
            path = asyncio.run(mp.create_single_chunk(src, i, 10, self.tmp, duration_seconds=duration))
            self.assertIsNotNone(path)
            total += decoded_duration(path)
            os.remove(path)
        self.assertAlmostEqual(total, decoded_duration(src), delta=0.2)

    def test_create_single_chunk_raises_on_ffmpeg_error_within_duration(self):
        bad = os.path.join(self.tmp, "broken.mp3")
        with open(bad, "wb") as f:
            f.write(b"not audio at all " * 500)
        mp = MediaProcessor()
        with self.assertRaises(Exception) as ctx:
            asyncio.run(mp.create_single_chunk(bad, 0, 10, self.tmp, duration_seconds=30))
        self.assertIn("Failed to create chunk 0", str(ctx.exception))
        # Also raises when the duration is unknown
        with self.assertRaises(Exception):
            asyncio.run(mp.create_single_chunk(bad, 1, 10, self.tmp))

    def test_create_single_chunk_returns_none_past_known_end(self):
        bad = os.path.join(self.tmp, "broken.mp3")
        with open(bad, "wb") as f:
            f.write(b"not audio at all " * 500)
        mp = MediaProcessor()
        self.assertIsNone(
            asyncio.run(mp.create_single_chunk(bad, 3, 10, self.tmp, duration_seconds=30))
        )


class VbrDurationTests(TempDirTestCase):
    def test_extract_audio_fixes_underestimated_vbr_duration(self):
        src = os.path.join(self.tmp, "vbr.mp3")
        make_vbr_without_xing(src)
        true_len = decoded_duration(src)
        raw_probe = probe_duration(src)
        # Reproduces the bug: header-less VBR is estimated far too short
        self.assertLess(raw_probe, true_len * 0.8, f"probe={raw_probe} decoded={true_len}")

        mp = MediaProcessor()
        audio = asyncio.run(mp.extract_audio(src))
        self.assertNotEqual(audio, src)
        self.assertTrue(os.path.exists(src), "original upload must be left for normal cleanup")

        duration, num_chunks = asyncio.run(mp.get_chunk_info(audio, chunk_duration=30))
        self.assertAlmostEqual(duration, true_len, delta=0.5)
        self.assertGreaterEqual(num_chunks * 30, true_len - mp_module.MIN_FINAL_CHUNK_SECONDS)
        self.assertEqual(num_chunks, 4)  # ~100s at 30s chunks; raw probe would plan 2

    def test_local_file_pipeline_transcribes_every_chunk(self):
        src = os.path.join(self.tmp, "vbr.mp3")
        make_vbr_without_xing(src)
        true_len = decoded_duration(src)
        tw_module.CHUNK_DURATION_SECONDS = 30
        worker = make_worker()
        chunk_lengths = []

        def reply(path, n):
            length = decoded_duration(path)
            chunk_lengths.append(length)
            # Transcript runs to the end of each chunk, so no tail recovery triggers
            return SimpleNamespace(
                text=f"chunk {n}",
                segments=[{"start": 0.0, "end": length, "text": f"chunk {n}"}],
            )

        api = install_fake_api(worker, reply)
        transcripts = asyncio.run(
            worker._process_local_file("job-local", {"local_file_path": src})
        )
        self.assertEqual(len(api.calls), 4)
        self.assertEqual(
            transcripts,
            ["[00:00:00] chunk 1", "[00:00:30] chunk 2", "[00:01:00] chunk 3", "[00:01:30] chunk 4"],
        )
        self.assertAlmostEqual(sum(chunk_lengths), true_len, delta=0.5)


class FakeB2Client:
    """Feeds a local file into the streaming pipe, pausing partway so that ffmpeg is
    still running while the first chunks are being transcribed."""

    def __init__(self, source_path, pause_after_fraction=0.3, pause_seconds=3.0):
        self.source_path = source_path
        self.pause_after_fraction = pause_after_fraction
        self.pause_seconds = pause_seconds

    async def download_to_stream(self, bucket_name, file_path, stream):
        import time

        def _stream():
            try:
                with open(self.source_path, "rb") as f:
                    data = f.read()
                cut = int(len(data) * self.pause_after_fraction)
                stream.write(data[:cut])
                stream.flush()
                time.sleep(self.pause_seconds)
                stream.write(data[cut:])
                stream.flush()
            finally:
                stream.close()

        await asyncio.get_event_loop().run_in_executor(None, _stream)


class StreamingRaceTests(TempDirTestCase):
    def test_every_streamed_segment_is_transcribed(self):
        src = os.path.join(self.tmp, "long.mp3")
        make_tone(src, 45)  # 10s segments -> chunk_000..chunk_004 (last one 5s)
        tw_module.CHUNK_DURATION_SECONDS = 10
        worker = make_worker()
        transcribed = []

        async def slow_transcribe(job_id, path, index, total):
            self.assertTrue(os.path.exists(path))
            transcribed.append(index)
            # Long enough for ffmpeg to finish the rest of the file and exit meanwhile
            await asyncio.sleep(1.5)
            return f"segment {index}"

        worker._transcribe_chunk_with_retry = slow_transcribe
        job = {"job_id": "job-stream", "b2_bucket": "bucket", "b2_file_path": "long.mp3"}
        results = asyncio.run(
            worker._stream_b2_to_transcription("job-stream", job, FakeB2Client(src))
        )
        self.assertEqual(sorted(transcribed), [0, 1, 2, 3, 4])
        self.assertEqual(results, [f"segment {i}" for i in range(5)])


class TailRecoveryTests(TempDirTestCase):
    def _chunk(self, seconds=60):
        path = os.path.join(self.tmp, "chunk_2.mp3")
        make_tone(path, seconds)
        return path

    def test_tail_is_retranscribed_once_and_merged(self):
        chunk = self._chunk(60)
        tw_module.CHUNK_DURATION_SECONDS = 600  # chunk index 2 -> offset 1200s
        worker = make_worker()
        tail_lengths = []

        def reply(path, n):
            if n == 1:
                # whisper-1 stopped 40s before the end of the chunk
                return SimpleNamespace(text="hello world", segments=[
                    {"start": 0.0, "end": 10.0, "text": " hello"},
                    {"start": 10.0, "end": 20.0, "text": " world"},
                ])
            tail_lengths.append(decoded_duration(path))
            return SimpleNamespace(text="...", segments=[
                # Overlap with the first pass (shifted end 19.9 <= 20): dropped
                {"start": 0.0, "end": 1.9, "text": " world"},
                {"start": 2.0, "end": 10.0, "text": " tail words", "no_speech_prob": 0.01, "avg_logprob": -0.2},
                # Hallucination on silence: dropped by the no-speech heuristic
                {"start": 30.0, "end": 40.0, "text": " Thanks for watching!", "no_speech_prob": 0.9, "avg_logprob": -1.5},
            ])

        api = install_fake_api(worker, reply)
        with self.assertLogs(tw_module.logger, level="WARNING") as logs:
            transcript = asyncio.run(worker._transcribe_chunk_with_retry("job-tail", chunk, 2, 3))

        self.assertEqual(len(api.calls), 2)
        self.assertEqual(api.calls[0]["path"], chunk)
        self.assertTrue(api.calls[1]["path"].endswith("chunk_2_tail.mp3"))
        self.assertTrue(all(c["model"] == "whisper-1" for c in api.calls))
        # Tail cut starts 2s before the last segment's end (18s) and runs to the end (60s)
        self.assertAlmostEqual(tail_lengths[0], 42.0, delta=0.3)
        self.assertFalse(os.path.exists(api.calls[1]["path"]), "tail file must be cleaned up")
        self.assertEqual(
            transcript.splitlines(),
            ["[00:20:00] hello", "[00:20:10] world", "[00:20:20] tail words"],
        )
        self.assertTrue(any("tail recovery" in line for line in logs.output))

    def test_empty_tail_result_is_fine(self):
        chunk = self._chunk(60)
        worker = make_worker()

        def reply(path, n):
            if n == 1:
                return SimpleNamespace(text="hi", segments=[{"start": 0.0, "end": 5.0, "text": "hi"}])
            return SimpleNamespace(text="", segments=[])

        api = install_fake_api(worker, reply)
        transcript = asyncio.run(worker._transcribe_chunk_with_retry("job-tail", chunk, 0, 1))
        self.assertEqual(len(api.calls), 2)
        self.assertEqual(transcript, "[00:00:00] hi")

    def test_no_recovery_when_transcript_reaches_chunk_end(self):
        chunk = self._chunk(60)
        worker = make_worker()

        def reply(path, n):
            return SimpleNamespace(text="all", segments=[{"start": 0.0, "end": 55.0, "text": "all"}])

        api = install_fake_api(worker, reply)
        transcript = asyncio.run(worker._transcribe_chunk_with_retry("job-tail", chunk, 0, 1))
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(transcript, "[00:00:00] all")


class UploadCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(dir=_TEMP_ROOT)
        self.manager = UploadManager(os.path.join(self.tmp, "uploads"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_complete_without_total_size_warns(self):
        session = self.manager.init("a.mp3", None)
        self.manager.append(session["upload_id"], 0, b"abc")
        with self.assertLogs("services.upload_manager", level=logging.WARNING):
            self.manager.complete(session["upload_id"], os.path.join(self.tmp, "a.mp3"))

    def test_truncated_upload_is_rejected_when_size_given_at_init(self):
        session = self.manager.init("a.mp3", 10)
        self.manager.append(session["upload_id"], 0, b"abc")
        with self.assertRaises(UploadError):
            self.manager.complete(session["upload_id"], os.path.join(self.tmp, "a.mp3"))


if __name__ == "__main__":
    unittest.main()
