"""Find media files in the paths given on the command line, and decode and chunk audio."""

from __future__ import annotations

import os
import pathlib
import subprocess  # nosec B404
from typing import TYPE_CHECKING, AbstractSet, Dict, Iterable, List, Tuple, Union

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

SAMPLE_RATE = 16000  # what every backend and pyannote expect

# Audio/video formats picked up when a folder is given (anything ffmpeg decodes also works as a file).
MEDIA_EXTENSIONS = frozenset(
    {".aac", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".oga", ".ogg", ".opus", ".wav", ".webm", ".wma"}
)


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


def load_audio(path: Union[str, "os.PathLike[str]"], sr: int = SAMPLE_RATE) -> "np.ndarray":
    """Decode any audio/video file to a mono float32 waveform at *sr* Hz with ffmpeg (as Whisper does)."""
    import numpy as np  # imported lazily: listing backends shouldn't pay for it

    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "0", "-i", str(path)]
    cmd += ["-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
    try:
        # A fixed argument list and no shell, so the file name can't inject commands.
        out = subprocess.run(cmd, capture_output=True, check=True).stdout  # nosec B603
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found: install it and make sure it is on your PATH") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode(errors='replace').strip()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


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
