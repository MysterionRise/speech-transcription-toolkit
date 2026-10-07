"""Find media files in the paths given on the command line, and decode and chunk audio."""

from __future__ import annotations

import os
import pathlib
import re
import subprocess  # nosec B404
import threading
from types import TracebackType
from typing import IO, TYPE_CHECKING, AbstractSet, Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from .errors import AudioDecodeError

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

SAMPLE_RATE = 16000  # what every backend and pyannote expect

# Audio/video formats picked up when a folder is given (anything ffmpeg decodes also works as a file).
MEDIA_EXTENSIONS = frozenset(
    {".aac", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".oga", ".ogg", ".opus", ".wav", ".webm", ".wma"}
)

# What load_audio() decodes: a path, the file's content, or a binary file object.
AudioSource = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview, IO[bytes]]

# The demuxers load_audio(strict=True) allows, as `ffmpeg -demuxers` names them: self-contained audio and video
# formats in common use. "mov" covers MP4, M4A and 3GP; "matroska" WebM and MKA; "asf" WMA and WMV; "ogg" Opus, Vorbis
# and Speex; "mpeg" and "mpegts" MPEG program and transport streams. Everything else is refused, notably the inputs
# that make ffmpeg open more files or URLs: HLS and DASH playlists, ffconcat lists, IMF and SDP.
STRICT_FORMATS = (
    "aac",
    "ac3",
    "aiff",
    "amr",
    "asf",
    "au",
    "avi",
    "caf",
    "eac3",
    "flac",
    "flv",
    "matroska",
    "mov",
    "mp3",
    "mpeg",
    "mpegts",
    "ogg",
    "w64",
    "wav",
)

# ffmpeg runs with only these environment variables, so secrets such as HF_TOKEN never reach it. A Windows program
# can't start without SYSTEMROOT and keeps temporary files in TEMP or TMP (elsewhere these are usually unset); the
# library paths let an ffmpeg that finds its shared libraries through them start.
_FFMPEG_ENV = ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SYSTEMROOT", "TEMP", "TMP")
_PIPE = "pipe:0"  # ffmpeg's name for its standard input
_STREAM_CHUNK_BYTES = 1024 * 1024  # how much of a file object is read at a time
_MAX_REASON_CHARS = 200  # ffmpeg's reason in an error message is cut to this length
_LOG_CONTEXT = re.compile(r"\[([^\]]*)\]\s*")  # the "[mp3 @ 0x55d5c8a0b2c0] " that starts many of ffmpeg's lines
_SUMMARY = re.compile(r"^Error opening (?:input|output) files?: ")  # ffmpeg 6+ ends with this; keep only the reason


def collect_files(
    paths: Iterable[pathlib.Path], extensions: AbstractSet[str]
) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """Expand *paths* into ``(file, relative_path)`` pairs, without duplicates.

    Folders are searched recursively for files whose extension (in any case) is in
    *extensions*; ``relative_path`` is the file's path inside that folder, so callers
    can mirror the folder tree in an output directory. Other paths are returned as-is,
    with their file name as ``relative_path``.
    """
    found: Dict[pathlib.Path, pathlib.Path] = {}
    for path in paths:
        if path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.is_file() and file.suffix.lower() in extensions:
                    found.setdefault(file, file.relative_to(path))
        else:
            found.setdefault(path, pathlib.Path(path.name))
    return list(found.items())


def load_audio(
    source: AudioSource,
    sr: int = SAMPLE_RATE,
    *,
    max_seconds: Optional[float] = None,
    timeout: Optional[float] = None,
    strict: bool = False,
) -> "np.ndarray":
    """Decode audio or video to a mono float32 waveform at *sr* Hz with ffmpeg (as Whisper does).

    Args:
        source: A path to anything ffmpeg decodes, the file's content as ``bytes``, or a binary file object, which is
            read from its current position in chunks: it is never held in memory whole, and it isn't closed. Bytes
            and file objects reach ffmpeg through its standard input (``pipe:0``). A pipe can't seek, so an MP4, M4A
            or MOV file whose index comes after its audio, as phones often record them, decodes only from a path.
        sr: Sample rate of the result.
        max_seconds: Decode at most this many seconds from the start (ffmpeg's ``-t``); the rest is ignored.
        timeout: Stop ffmpeg and raise :class:`AudioDecodeError` if decoding takes longer than this many seconds.
        strict: For untrusted input. ffmpeg may then use only the demuxers in :data:`STRICT_FORMATS`, the common
            self-contained audio and video formats (its ``-format_whitelist``), so playlists (HLS, DASH), ffconcat
            lists and any other input that makes ffmpeg open more files or URLs are refused. An allowlist, because
            ffmpeg has no option to refuse given demuxers, and spotting such inputs ourselves would miss some: one
            inside another format, or a demuxer that a later ffmpeg adds. ffmpeg applies the allowlist to whatever a
            demuxer opens, too.

    ffmpeg opens only local files and its standard input, never a URL (``-protocol_whitelist file,pipe``); a path is
    always read as a local file, even one that looks like a URL (``10:30.mp3``). ffmpeg runs without the caller's
    environment, so secrets such as ``HF_TOKEN`` don't reach it: it gets only ``PATH``, the shared-library search
    paths, and the ``SYSTEMROOT``, ``TEMP`` and ``TMP`` that Windows programs need.

    Memory: ffmpeg's output (2 bytes per sample) and the returned array (4) are the only copies, so decoding peaks at
    about 6 bytes per sample, some 1 GB for three hours.

    Returns:
        The samples as a writable float32 array in [-1, 1).

    Raises:
        AudioDecodeError: If ffmpeg is missing, takes longer than *timeout*, or can't decode the input, including a
            missing file and, with *strict*, a format that isn't allowed. The message ends with ffmpeg's reason,
            without the input's path or ffmpeg's banner. It is also a ``RuntimeError``.
        TypeError: If *source* isn't a path, bytes or a binary file object.
        ValueError: If *max_seconds* or *timeout* isn't a positive number.
        OSError: If reading the file object fails (whatever else its ``read()`` raises propagates too); ffmpeg's
            result is discarded then, since its input was cut short.
    """
    import numpy as np  # imported lazily: listing backends shouldn't pay for it

    for name, value in (("max_seconds", max_seconds), ("timeout", timeout)):
        if value is not None and not value > 0:
            raise ValueError(f"{name} must be a positive number of seconds, got {value!r}")
    with _FfmpegInput(source) as ffmpeg_input:
        cmd = _ffmpeg_command(ffmpeg_input.url, sr, max_seconds, strict)
        done = _run_ffmpeg(cmd, timeout, ffmpeg_input)
    stderr = done.stderr or b""
    # ffmpeg can also exit successfully having decoded nothing, such as an MP4 whose index it couldn't seek back to.
    if done.returncode != 0 or (not done.stdout and stderr.strip()):
        reason = _failure_reason(stderr, done.returncode, ffmpeg_input.url)
        raise AudioDecodeError(f"Failed to load audio: {reason}")
    audio = np.frombuffer(done.stdout, np.int16).astype(np.float32)
    audio /= 32768.0  # in place: ffmpeg's output and this array are the only copies
    return audio


class _FfmpegInput:
    """How a source reaches ffmpeg: the name ffmpeg opens (``url``), and the ``input`` or ``stdin`` it gets.

    ffmpeg opens a path itself; ``subprocess.run`` writes bytes to its stdin; a thread copies a file object to its
    stdin in chunks, until the end of the file or until ffmpeg stops reading. Leaving the ``with`` block waits for the
    thread, and raises what reading the file object raised.
    """

    def __init__(self, source: AudioSource) -> None:
        self.url = _PIPE
        self.input: Union[bytes, memoryview, None] = None
        self.stdin: Optional[int] = None
        self._stream: Any = None
        self._feeder: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None
        if isinstance(source, bytes):
            self.input = source
        elif isinstance(source, (bytearray, memoryview)):
            self.input = memoryview(source).cast("B")  # bytes, whatever the item type: the pipe counts bytes
        elif isinstance(source, (str, os.PathLike)):
            self.url = "file:" + os.fsdecode(source)  # a local file, even if the path looks like a URL
            self.stdin = subprocess.DEVNULL
        elif callable(getattr(source, "read", None)):
            self._stream = source
        else:
            raise TypeError(f"load_audio() takes a path, bytes or a binary file object, not {type(source).__name__}")

    def __enter__(self) -> "_FfmpegInput":
        if self._stream is not None:
            self.stdin, write_end = os.pipe()
            self._feeder = threading.Thread(target=self._feed, args=(write_end,), name="load_audio", daemon=True)
            self._feeder.start()
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]
    ) -> None:
        if self._feeder is not None and self.stdin is not None:
            os.close(self.stdin)  # ffmpeg has exited, so a write blocked on the full pipe fails now
            self._feeder.join()
        # The input was cut short, so whatever ffmpeg made of it is wrong; but Ctrl-C still stops the program.
        if self._error is not None and (exc_type is None or issubclass(exc_type, Exception)):
            raise self._error

    def _feed(self, write_end: int) -> None:
        """Copy the file object to ffmpeg's stdin, then close the pipe so that ffmpeg sees the end of the input."""
        try:
            while True:
                try:
                    chunk = self._read_chunk()
                except Exception as e:  # the caller's stream failed, or isn't binary
                    self._error = e
                    return
                if not chunk:
                    return
                try:
                    while chunk:
                        chunk = chunk[os.write(write_end, chunk) :]
                except OSError:  # ffmpeg stopped reading: it has finished, failed or been stopped (see its status)
                    return
        finally:
            os.close(write_end)

    def _read_chunk(self) -> memoryview:
        chunk = self._stream.read(_STREAM_CHUNK_BYTES)
        if chunk and not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError(f"load_audio() needs a binary file object, but read() returned {type(chunk).__name__}")
        return memoryview(chunk or b"").cast("B")


def _ffmpeg_command(url: str, sr: int, max_seconds: Optional[float], strict: bool) -> List[str]:
    """The ffmpeg command that decodes *url* to mono 16-bit PCM at *sr* Hz on stdout."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "0"]
    # Input options: only local files and the pipe, perhaps only common formats, and perhaps only the first seconds.
    cmd += ["-protocol_whitelist", "file,pipe"]
    if strict:
        cmd += ["-format_whitelist", ",".join(STRICT_FORMATS)]
    if max_seconds is not None:
        cmd += ["-t", f"{max_seconds:.6f}".rstrip("0").rstrip(".")]  # ffmpeg refuses exponents such as 1e-05
    cmd += ["-i", url, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
    return cmd


def _run_ffmpeg(
    cmd: List[str], timeout: Optional[float], ffmpeg_input: _FfmpegInput
) -> "subprocess.CompletedProcess[bytes]":
    """Run ffmpeg with a minimal environment; a missing ffmpeg and a timeout raise AudioDecodeError."""
    env = {name: os.environ[name] for name in _FFMPEG_ENV if name in os.environ}
    try:
        # A fixed argument list and no shell, so the input can't inject commands.
        return subprocess.run(  # nosec B603
            cmd,
            input=ffmpeg_input.input,
            stdin=ffmpeg_input.stdin,
            capture_output=True,
            env=env,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise AudioDecodeError("ffmpeg not found: install it and make sure it is on your PATH") from e
    except subprocess.TimeoutExpired as e:
        raise AudioDecodeError(f"Failed to load audio: decoding took longer than the {e.timeout:g} s timeout") from e


def _failure_reason(stderr: bytes, returncode: int, url: str) -> str:
    """Why ffmpeg failed, in one line: a known problem in our words, else ffmpeg's last line, made safe to show."""
    messages = _ffmpeg_messages(stderr)
    for context, message in messages:
        if message.startswith("Format not on whitelist"):  # ffmpeg names the demuxer as the message's context
            return f"the input's format ({context or 'unknown'}) isn't allowed in strict mode"
        if "does not contain any stream" in message:
            return "the input has no audio stream"
        if "partial file" in message and url == _PIPE:
            return (
                "this MP4, M4A or MOV file has its index at the end, so it can only be decoded from a path, "
                "not from bytes or a stream"
            )
        if "not on whitelist" in message:  # a playlist that refers to a URL, say
            return _shown(message, url)
    if not messages:
        return f"ffmpeg exited with status {returncode}"
    return _shown(_SUMMARY.sub("", messages[-1][1]), url)


def _ffmpeg_messages(stderr: bytes) -> List[Tuple[str, str]]:
    """ffmpeg's error lines as ``(context, message)``: "[hls @ 0x5d] Format..." gives ``("hls", "Format...")``."""
    messages = []
    # Decoded as os.fsdecode() decodes paths, so that a path ffmpeg prints matches the one it was given.
    for line in stderr.decode("utf-8", "surrogateescape").splitlines():
        line, context = line.strip(), ""
        while match := _LOG_CONTEXT.match(line):
            context = context or match.group(1).split(" @ ")[0]
            line = line[match.end() :]
        if line and not line.startswith("Last message repeated"):
            messages.append((context, line))
    return messages


def _shown(message: str, url: str) -> str:
    """A line of ffmpeg's output, fit for an error message: printable, short, and without the input's path.

    The path may be a temporary file's, which callers such as a server shouldn't reveal.
    """
    message = message.removeprefix(f"{url}: ").replace(url, "the input")  # ffmpeg before 6.0: "<input>: <reason>"
    message = "".join(char if char.isprintable() else "?" for char in message)
    if len(message) > _MAX_REASON_CHARS:
        message = message[: _MAX_REASON_CHARS - 3] + "..."
    return message


def split_audio(
    audio: "np.ndarray", max_seconds: float, sr: int = SAMPLE_RATE, search_seconds: float = 5.0
) -> List[Tuple[float, "np.ndarray"]]:
    """Cut *audio* into chunks of at most *max_seconds*, as ``(start time in seconds, samples)`` pairs.

    Each cut goes in the quietest 100 ms of the last *search_seconds* before the limit, so words
    are rarely chopped in half.
    """
    import numpy as np

    max_len = int(max_seconds * sr)
    window = int(0.1 * sr)
    search = min(int(search_seconds * sr), max_len // 2)
    chunks: List[Tuple[float, "np.ndarray"]] = []
    start = 0
    while len(audio) - start > max_len:
        region = audio[start + max_len - search : start + max_len]
        frames = len(region) // window
        energy = np.square(region[: frames * window]).reshape(frames, window).mean(axis=1)
        cut = start + max_len - search + int(np.argmin(energy)) * window + window // 2
        chunks.append((start / sr, audio[start:cut]))
        start = cut
    if len(audio) > start:
        chunks.append((start / sr, audio[start:]))
    return chunks
