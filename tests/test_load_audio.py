"""Tests for speech_toolkit.media.load_audio: ffmpeg's arguments, limits, environment, errors and memory use.

subprocess.run is mocked, so these tests need no ffmpeg. TestWithFfmpeg also runs the real one when it is installed.
"""

from __future__ import annotations

import io
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import tracemalloc
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from speech_toolkit import AudioDecodeError
from speech_toolkit.media import load_audio

PCM = np.array([0, 16384, -32768], dtype=np.int16).tobytes()  # what ffmpeg writes for 0.0, 0.5 and -1.0
FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "0"]
PROTOCOLS = ["-protocol_whitelist", "file,pipe"]  # always: local files and the pipe, never a URL
STRICT = ["-format_whitelist", "aac,ac3,aiff,amr,asf,au,avi,caf,eac3,flac,flv,matroska,mov,mp3,mpeg,mpegts,ogg,w64,wav"]
OUTPUT = ["-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-"]
CHUNK = 1024 * 1024  # how much load_audio reads from a file object at a time
TEMP_FILE = "/tmp/tmpk2j3x9.wav"  # a path that error messages mustn't reveal


def finished(stdout: bytes = PCM, stderr: bytes = b"", returncode: int = 0) -> subprocess.CompletedProcess:
    """What subprocess.run returns when ffmpeg exits."""
    return subprocess.CompletedProcess(args=["ffmpeg"], returncode=returncode, stdout=stdout, stderr=stderr)


def reading_stdin(received: bytearray, result: Optional[subprocess.CompletedProcess] = None, limit: int = -1):
    """A fake subprocess.run that, like ffmpeg, reads its stdin (to the end, or *limit* bytes) before it exits."""

    def run(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        while limit < 0 or len(received) < limit:
            chunk = os.read(kwargs["stdin"], 65536)
            if not chunk:
                break
            received.extend(chunk)
        return result or finished()

    return run


def within(seconds: float, call: Callable[[], Any]) -> Any:
    """call(), but fail the test instead of hanging the run if it doesn't return in time."""
    outcome: Dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = call()
        except BaseException as e:  # handed to the test's thread
            outcome["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        pytest.fail(f"load_audio() didn't return within {seconds} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


class Recording(io.BytesIO):
    """A binary file object that records the size of each read()."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.sizes: List[Optional[int]] = []

    def read(self, size: Optional[int] = -1) -> bytes:
        self.sizes.append(size)
        return super().read(size)


class FailingStream(io.RawIOBase):
    """A binary stream whose second read fails, like an interrupted upload."""

    def __init__(self) -> None:
        self.reads = 0

    def readable(self) -> bool:
        return True

    def read(self, size: Optional[int] = -1) -> bytes:
        self.reads += 1
        if self.reads > 1:
            raise ConnectionResetError("upload interrupted")
        return b"RIFF" + bytes(100)


@pytest.fixture
def ffmpeg():
    """The mocked subprocess.run: by default, ffmpeg decodes three samples."""
    with patch("speech_toolkit.media.subprocess.run") as run:
        run.return_value = finished()
        yield run


def _sources() -> Dict[str, Any]:
    return {"path": "talk.mp3", "bytes": b"RIFF....WAVE", "file object": io.BytesIO(b"RIFF....WAVE")}


class TestCommand:
    @pytest.mark.parametrize("kind, name", [("path", "file:talk.mp3"), ("bytes", "pipe:0"), ("file object", "pipe:0")])
    @pytest.mark.parametrize("strict", [False, True])
    @pytest.mark.parametrize("max_seconds, limit", [(None, []), (30, ["-t", "30"]), (2.5, ["-t", "2.5"])])
    def test_exact_arguments(self, ffmpeg, kind, name, strict, max_seconds, limit):
        load_audio(_sources()[kind], max_seconds=max_seconds, strict=strict)

        expected = FFMPEG + PROTOCOLS + (STRICT if strict else []) + limit + ["-i", name] + OUTPUT
        assert ffmpeg.call_args.args == (expected,)

    def test_path_input(self, ffmpeg):
        load_audio(pathlib.Path("2024-05-01T10:30.mp3"))

        cmd, kwargs = ffmpeg.call_args.args[0], ffmpeg.call_args.kwargs
        # "file:" keeps ffmpeg from taking "2024-05-01T10" for a protocol; ffmpeg doesn't get the caller's stdin.
        assert cmd[cmd.index("-i") + 1] == "file:2024-05-01T10:30.mp3"
        assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["input"] is None
        assert kwargs["capture_output"] is True

    def test_windows_path(self, ffmpeg):
        load_audio(pathlib.PureWindowsPath(r"C:\Users\me\talk.mp3"))

        cmd = ffmpeg.call_args.args[0]
        assert cmd[cmd.index("-i") + 1] == r"file:C:\Users\me\talk.mp3"

    def test_bytes_input(self, ffmpeg):
        data = b"RIFF....WAVE"
        load_audio(data)

        assert ffmpeg.call_args.kwargs["input"] is data and ffmpeg.call_args.kwargs["stdin"] is None

    @pytest.mark.parametrize("wrap", [bytearray, memoryview, lambda data: memoryview(np.frombuffer(data, np.int16))])
    def test_bytes_like_input_is_sent_byte_for_byte(self, ffmpeg, wrap):
        data = bytes(range(8))
        load_audio(wrap(data))

        assert bytes(ffmpeg.call_args.kwargs["input"]) == data

    def test_sample_rate(self, ffmpeg):
        load_audio("talk.mp3", sr=8000)

        cmd = ffmpeg.call_args.args[0]
        assert cmd[cmd.index("-ar") + 1] == "8000"

    @pytest.mark.parametrize(
        "seconds, arg", [(0.00001, "0.00001"), (3600, "3600"), (0.1 + 0.2, "0.3"), (1e5, "100000")]
    )
    def test_max_seconds_never_uses_exponents(self, ffmpeg, seconds, arg):
        load_audio("talk.mp3", max_seconds=seconds)

        cmd = ffmpeg.call_args.args[0]
        assert cmd[cmd.index("-t") + 1] == arg


class TestFileObjects:
    def test_streamed_in_chunks_from_the_current_position(self, ffmpeg):
        data = os.urandom(3 * CHUNK + 123)
        stream = Recording(b"header" + data)
        stream.read(6)  # the caller already read a header
        stream.sizes.clear()
        received = bytearray()
        ffmpeg.side_effect = reading_stdin(received)

        audio = load_audio(stream)

        assert audio.tolist() == [0.0, 0.5, -1.0]
        assert received == data
        assert stream.sizes == [CHUNK] * 5  # never read whole: four chunks, then the end of the file
        assert not stream.closed  # the caller's to close
        assert isinstance(ffmpeg.call_args.kwargs["stdin"], int) and ffmpeg.call_args.kwargs["input"] is None

    def test_ffmpeg_stopping_early_ends_the_copy(self, ffmpeg):
        """ffmpeg stops reading (bad data, or max_seconds reached) with most of a big file still to come."""
        received = bytearray()
        stderr = b"[in#0 @ 0x55f6] Error opening input: Invalid data found when processing input\n"
        ffmpeg.side_effect = reading_stdin(received, finished(b"", stderr, 183), limit=1000)

        with pytest.raises(AudioDecodeError, match="Invalid data found when processing input"):
            within(60, lambda: load_audio(io.BytesIO(bytes(8 * CHUNK))))

        assert len(received) >= 1000
        assert not [thread for thread in threading.enumerate() if thread.name == "load_audio"]

    def test_stream_errors_are_raised(self, ffmpeg):
        """An interrupted stream must not pass for a short recording, even when ffmpeg decodes what came."""
        ffmpeg.side_effect = reading_stdin(bytearray())

        with pytest.raises(ConnectionResetError, match="upload interrupted"):
            load_audio(FailingStream())

    def test_stream_errors_win_over_ffmpeg_errors(self, ffmpeg):
        ffmpeg.side_effect = reading_stdin(bytearray(), finished(b"", b"pipe:0: Invalid data found\n", 1))

        with pytest.raises(ConnectionResetError, match="upload interrupted"):
            load_audio(FailingStream())

    def test_stream_errors_win_over_a_timeout(self, ffmpeg):
        def timed_out(cmd, **kwargs):
            reading_stdin(bytearray())(cmd, **kwargs)
            raise subprocess.TimeoutExpired(cmd, 5)

        ffmpeg.side_effect = timed_out

        with pytest.raises(ConnectionResetError) as raised:
            load_audio(FailingStream(), timeout=5)

        assert isinstance(raised.value.__context__, AudioDecodeError)  # the timeout isn't lost

    def test_ctrl_c_wins_over_stream_errors(self, ffmpeg):
        def interrupted(cmd, **kwargs):
            reading_stdin(bytearray())(cmd, **kwargs)  # the stream has failed by the time ffmpeg sees its end
            raise KeyboardInterrupt

        ffmpeg.side_effect = interrupted

        with pytest.raises(KeyboardInterrupt):
            load_audio(FailingStream())

    def test_text_file_object_is_refused(self, ffmpeg):
        ffmpeg.side_effect = reading_stdin(bytearray())

        with pytest.raises(TypeError, match="needs a binary file object, but read.. returned str"):
            load_audio(io.StringIO("not audio"))

    def test_open_file(self, ffmpeg, tmp_path):
        path = tmp_path / "talk.wav"
        path.write_bytes(b"RIFF" + os.urandom(5000))
        received = bytearray()
        ffmpeg.side_effect = reading_stdin(received)

        with path.open("rb") as file:
            load_audio(file)

        assert received == path.read_bytes()


@pytest.mark.parametrize("source", [42, None, 1.5, np.zeros(3, dtype=np.float32)])
def test_other_sources_are_refused(ffmpeg, source):
    with pytest.raises(TypeError, match="takes a path, bytes or a binary file object"):
        load_audio(source)
    ffmpeg.assert_not_called()


class TestLimits:
    def test_timeout_goes_to_subprocess(self, ffmpeg):
        load_audio("talk.mp3", timeout=12.5)
        assert ffmpeg.call_args.kwargs["timeout"] == 12.5

        load_audio("talk.mp3")
        assert ffmpeg.call_args.kwargs["timeout"] is None

    @pytest.mark.parametrize("kind", ["path", "bytes", "file object"])
    def test_timeout_raises_audio_decode_error(self, ffmpeg, kind):
        ffmpeg.side_effect = subprocess.TimeoutExpired(["ffmpeg", "-i", f"file:{TEMP_FILE}"], 5)

        with pytest.raises(AudioDecodeError) as raised:
            load_audio(_sources()[kind], timeout=5)

        assert str(raised.value) == "Failed to load audio: decoding took longer than the 5 s timeout"

    @pytest.mark.parametrize("option", ["max_seconds", "timeout"])
    @pytest.mark.parametrize("value", [0, -1, float("nan")])
    def test_limits_must_be_positive(self, ffmpeg, option, value):
        with pytest.raises(ValueError, match=f"{option} must be a positive number of seconds"):
            load_audio("talk.mp3", **{option: value})
        ffmpeg.assert_not_called()


class TestEnvironment:
    SECRETS = {
        "HF_TOKEN": "hf_secret",
        "HUGGINGFACE_TOKEN": "hf_old_secret",
        "TRANSCRIBE_API_KEY": "api_secret",
        "AWS_SECRET_ACCESS_KEY": "aws_secret",
    }

    @pytest.mark.parametrize("kind", ["path", "bytes", "file object"])
    def test_secrets_dont_reach_ffmpeg(self, ffmpeg, monkeypatch, kind):
        for name, value in self.SECRETS.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("PATH", os.pathsep.join(["/opt/ffmpeg/bin", "/usr/bin"]))

        load_audio(_sources()[kind])

        env = ffmpeg.call_args.kwargs["env"]
        assert "HF_TOKEN" not in env
        assert not set(env) & set(self.SECRETS)
        assert not [value for value in env.values() if "secret" in value]
        assert env["PATH"] == os.pathsep.join(["/opt/ffmpeg/bin", "/usr/bin"])
        assert set(env) <= {"PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SYSTEMROOT", "TEMP", "TMP"}

    def test_what_ffmpeg_needs_to_start_is_kept(self, ffmpeg, monkeypatch):
        needed = {
            "PATH": "/usr/bin",
            "LD_LIBRARY_PATH": "/opt/ffmpeg/lib",
            "SYSTEMROOT": r"C:\Windows",  # Windows can't start a program without it
            "TEMP": r"C:\Temp",
            "TMP": r"C:\Temp",
        }
        for name, value in needed.items():
            monkeypatch.setenv(name, value)

        load_audio("talk.mp3")

        env = ffmpeg.call_args.kwargs["env"]
        assert {name: env.get(name) for name in needed} == needed


class TestErrors:
    def test_missing_ffmpeg(self, ffmpeg):
        ffmpeg.side_effect = FileNotFoundError(2, "No such file or directory", "ffmpeg")

        with pytest.raises(AudioDecodeError, match="^ffmpeg not found: install it and make sure it is on your PATH$"):
            load_audio("talk.mp3")

    @pytest.mark.parametrize(
        "stderr, reason",
        [
            pytest.param(
                b"[in#0 @ 0x558d72719f00] Error opening input: No such file or directory\n"
                b"Error opening input file file:/tmp/tmpk2j3x9.wav.\n"
                b"Error opening input files: No such file or directory\n",
                "No such file or directory",
                id="ffmpeg 6+",
            ),
            pytest.param(
                b"file:/tmp/tmpk2j3x9.wav: Invalid data found when processing input\n",
                "Invalid data found when processing input",
                id="ffmpeg 4 and 5",
            ),
            pytest.param(
                b"[hls @ 0x55f487679040] Format not on whitelist 'aac,ac3,aiff,amr,asf,au,avi,caf,eac3,flac'\n"
                b"[in#0 @ 0x55f487678f40] Error opening input: Invalid argument\n"
                b"Error opening input file file:/tmp/tmpk2j3x9.wav.\n"
                b"Error opening input files: Invalid argument\n",
                "the input's format (hls) isn't allowed in strict mode",
                id="strict refuses a playlist",
            ),
            pytest.param(
                b"[out#0/s16le @ 0x562bed2defc0] Output file does not contain any stream\n"
                b"Error opening output file -.\n"
                b"Error opening output files: Invalid argument\n",
                "the input has no audio stream",
                id="no audio",
            ),
            pytest.param(
                b"[http @ 0x55df597f8540] Protocol 'http' not on whitelist 'file,pipe'!\n"
                b"[hls @ 0x55df59837000] Error when loading first segment 'http://example.com/seg.ts'\n"
                b"[in#0 @ 0x55df59836f00] Error opening input: Invalid data found when processing input\n"
                b"Error opening input file file:/tmp/tmpk2j3x9.wav.\n"
                b"Error opening input files: Invalid data found when processing input\n",
                "Protocol 'http' not on whitelist 'file,pipe'!",
                id="playlist refers to a URL",
            ),
            pytest.param(
                b"[mp3 @ 000001F2C3D4E5F0] Header missing\n    Last message repeated 2 times\n",
                "Header missing",
                id="Windows log context, repeats",
            ),
            pytest.param(
                b"[aist#0:0/mp3 @ 0x1] [dec:mp3 @ 0x2] Could not open file:/tmp/tmpk2j3x9.wav again\n",
                "Could not open the input again",
                id="path inside a line",
            ),
            pytest.param(b"Bad \x1b[31mred\x07 data\n", "Bad ?[31mred? data", id="control characters"),
            pytest.param(b"", "ffmpeg exited with status 1", id="no message"),
        ],
    )
    def test_reason_is_ffmpegs_without_paths(self, ffmpeg, stderr, reason):
        ffmpeg.return_value = finished(b"", stderr, returncode=1)

        with pytest.raises(AudioDecodeError) as raised:
            load_audio(TEMP_FILE)

        assert str(raised.value) == f"Failed to load audio: {reason}"
        assert "tmpk2j3x9" not in str(raised.value)

    def test_long_reasons_are_cut(self, ffmpeg):
        ffmpeg.return_value = finished(b"", b"x" * 1000 + b"\n", returncode=1)

        with pytest.raises(AudioDecodeError) as raised:
            load_audio("talk.mp3")

        reason = str(raised.value).removeprefix("Failed to load audio: ")
        assert len(reason) == 200 and reason.endswith("...")

    def test_killed_ffmpeg(self, ffmpeg):
        ffmpeg.return_value = finished(PCM, b"", returncode=-9)

        with pytest.raises(AudioDecodeError, match="^Failed to load audio: ffmpeg exited with status -9$"):
            load_audio("talk.mp3")

    MOOV_AT_END = (
        b"[mov,mp4,m4a,3gp,3g2,mj2 @ 0x564c72fbbfc0] stream 0, offset 0x2c: partial file\n"
        b"    Last message repeated 1 times\n"
        b"[in#0/mov,mp4,m4a,3gp,3g2,mj2 @ 0x1] Error during demuxing: Invalid data found when processing input\n"
        b"[in#0/mov,mp4,m4a,3gp,3g2,mj2 @ 0x1] Error retrieving a packet from demuxer: Invalid data found when "
        b"processing input\n"
    )

    @pytest.mark.parametrize("kind", ["bytes", "file object"])
    def test_mp4_with_its_index_at_the_end_through_a_pipe(self, ffmpeg, kind):
        """ffmpeg exits successfully having decoded nothing: it can't seek back to the audio in a pipe."""
        ffmpeg.return_value = finished(b"", self.MOOV_AT_END, returncode=0)

        with pytest.raises(AudioDecodeError, match="index at the end, so it can only be decoded from a path"):
            load_audio(_sources()[kind])

    def test_no_audio_and_errors_from_a_path(self, ffmpeg):
        ffmpeg.return_value = finished(b"", self.MOOV_AT_END, returncode=0)

        with pytest.raises(AudioDecodeError) as raised:
            load_audio("talk.m4a")

        assert str(raised.value) == (
            "Failed to load audio: Error retrieving a packet from demuxer: Invalid data found when processing input"
        )

    def test_no_audio_without_errors_is_empty(self, ffmpeg):
        ffmpeg.return_value = finished(b"", b"")

        audio = load_audio("empty.wav")

        assert audio.dtype == np.float32 and audio.shape == (0,)

    def test_audio_is_kept_despite_errors(self, ffmpeg):
        """ffmpeg skips a damaged frame and goes on: the rest of the recording is worth transcribing."""
        ffmpeg.return_value = finished(PCM, b"[mp3float @ 0x55] Header missing\n")

        assert load_audio("talk.mp3").tolist() == [0.0, 0.5, -1.0]

    def test_missing_file_is_still_a_runtime_error(self, ffmpeg):
        """Callers check for missing files themselves; load_audio() raised RuntimeError for them before."""
        ffmpeg.return_value = finished(b"", b"file:gone.mp3: No such file or directory\n", returncode=254)

        with pytest.raises(RuntimeError, match="^Failed to load audio: No such file or directory$"):
            load_audio("gone.mp3")


class TestOutput:
    def test_16_bit_pcm_becomes_float32(self, ffmpeg):
        audio = load_audio("talk.mp3")

        assert audio.dtype == np.float32
        assert audio.tolist() == [0.0, 0.5, -1.0]
        assert audio.flags.writeable and audio.flags.c_contiguous  # torch.from_numpy() takes it as is

    def test_peak_memory_is_about_6_bytes_per_sample(self, ffmpeg):
        """ffmpeg's output (2 bytes per sample) and the float32 result (4) are the only copies; it used to be 10-12."""
        samples = 2_000_000
        # Allocated while tracing, like the output that subprocess.run collects from ffmpeg.
        ffmpeg.side_effect = lambda cmd, **kwargs: finished(os.urandom(2 * samples))
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before, _ = tracemalloc.get_traced_memory()
            audio = load_audio("long.wav")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not was_tracing:
                tracemalloc.stop()

        assert len(audio) == samples
        assert 6 <= (peak - before) / samples < 6.5


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _has_ffmpeg(), reason="needs ffmpeg on PATH")
class TestWithFfmpeg:
    """The real ffmpeg accepts these arguments and decodes paths, bytes and file objects alike."""

    @pytest.fixture
    def tone(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Two seconds of a 440 Hz tone, as a 44.1 kHz WAV file."""
        path = tmp_path / "tone.wav"
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
        subprocess.run(cmd + ["-ar", "44100", str(path)], check=True, capture_output=True)
        return path

    def test_path_bytes_and_file_objects_decode_alike(self, tone):
        from_path = load_audio(tone)

        assert len(from_path) == 2 * 16000 and 0.05 < np.abs(from_path).max() < 1
        np.testing.assert_array_equal(load_audio(str(tone)), from_path)
        np.testing.assert_array_equal(load_audio(tone.read_bytes()), from_path)
        np.testing.assert_array_equal(load_audio(io.BytesIO(tone.read_bytes())), from_path)
        with tone.open("rb") as file:
            np.testing.assert_array_equal(load_audio(file), from_path)

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows file names can't contain ':'")
    def test_path_that_looks_like_a_url(self, tone):
        assert len(load_audio(tone.rename(tone.with_name("10:30.wav")))) == 2 * 16000

    def test_max_seconds(self, tone):
        assert len(load_audio(tone, max_seconds=0.5)) == 8000
        assert len(load_audio(tone.read_bytes(), max_seconds=0.5)) == 8000
        assert len(load_audio(io.BytesIO(tone.read_bytes()), max_seconds=0.5)) == 8000

    def test_strict_allows_common_formats_and_refuses_concat_lists(self, tone):
        assert len(load_audio(tone, strict=True)) == 2 * 16000
        assert len(load_audio(tone.read_bytes(), strict=True)) == 2 * 16000
        concat = tone.with_name("list.ffconcat")
        concat.write_text(f"ffconcat version 1.0\nfile '{tone.name}'\n", encoding="utf-8")

        with pytest.raises(AudioDecodeError, match=r"format \(concat\) isn't allowed in strict mode"):
            load_audio(concat.read_bytes(), strict=True)
        with pytest.raises(AudioDecodeError, match=r"format \(concat\) isn't allowed in strict mode"):
            load_audio(concat, strict=True)

    @pytest.mark.skipif(sys.platform == "win32", reason="ffmpeg resolves the list's entries as URLs, with '/'")
    def test_local_concat_list_without_strict(self, tone):
        concat = tone.with_name("list.ffconcat")
        concat.write_text(f"ffconcat version 1.0\nfile '{tone.name}'\nfile '{tone.name}'\n", encoding="utf-8")

        assert len(load_audio(concat)) == 4 * 16000  # strict is opt-in: the CLI still reads local lists

    def test_errors(self, tmp_path):
        with pytest.raises(AudioDecodeError, match="No such file or directory"):
            load_audio(tmp_path / "missing.wav")
        with pytest.raises(AudioDecodeError, match="^Failed to load audio: ") as raised:
            load_audio(b"this is not audio" * 100)
        assert "pipe:0" not in str(raised.value)
