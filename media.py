"""Find media files in the paths given on the command line."""

from __future__ import annotations

import pathlib
from typing import AbstractSet, Dict, Iterable, List, Tuple

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
