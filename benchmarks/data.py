"""The evaluation set: the clips that manifest.json lists, downloaded once into a cache folder and checked.

For each clip, the manifest names where its audio is (a file of a Hugging Face dataset at a fixed revision, and the
row of a Parquet file or the member of a tar archive in it), the SHA-256 of that audio, its length and its reference
transcript. Only the clips are downloaded: the rows of the Parquet files that hold them, and the start of each tar
archive. Every clip is checked against its SHA-256 before it is stored, and again before each run. Only the clips'
audio is kept, in the cache folder::

    python -m benchmarks.data                   # download the missing clips, check them all
    python -m benchmarks.data --data-dir DIR    # elsewhere than BENCHMARK_DATA_DIR or ~/.cache/speech-toolkit-benchmarks

``python -m benchmarks.run`` does the same before it runs. Downloading needs ``huggingface_hub`` (which faster-whisper
installs) and, for the Parquet files, ``pyarrow`` (``pip install -r requirements-bench.txt``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
import tarfile
import time
from typing import IO, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
DATA_DIR_VARIABLE = "BENCHMARK_DATA_DIR"
ATTEMPTS = 3  # downloads of one dataset file, before giving up
INSTALL_HINT = "pip install -r requirements-bench.txt"

Log = Callable[[str], None]


class DataError(RuntimeError):
    """The evaluation set can't be read, downloaded or checked."""


def log_to_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclasses.dataclass(frozen=True)
class Clip:
    """One recording of the evaluation set, and its reference transcript."""

    id: str  # unique in the manifest, e.g. "librispeech/1089-134686-0000"
    set: str  # the set it belongs to (a key of the manifest's "sets"): its results are added up per set
    language: str  # ISO 639-1 code, passed to the backend and choosing the text normalizer
    file: str  # the dataset file that holds its audio
    audio: str  # where it is stored in the cache folder (a relative path)
    sha256: str
    seconds: float
    text: str
    row: Optional[int] = None  # its row, for a Parquet file
    member: Optional[str] = None  # its member name, for a tar archive


@dataclasses.dataclass(frozen=True)
class Manifest:
    sets: Dict[str, Dict[str, Any]]  # per set: "title", "dataset", "revision", "license", ...
    clips: Tuple[Clip, ...]

    def select(self, sets: Optional[Sequence[str]] = None, limit: Optional[int] = None) -> List[Clip]:
        """The clips of *sets* (all by default), at most the first *limit* of each set."""
        unknown = sorted(set(sets or ()) - set(self.sets))
        if unknown:
            raise DataError(f"no set named {', '.join(unknown)} in the manifest; sets: {', '.join(sorted(self.sets))}")
        chosen: List[Clip] = []
        taken: Dict[str, int] = {}
        for clip in self.clips:
            if sets and clip.set not in sets:
                continue
            if limit is not None and taken.get(clip.set, 0) >= limit:
                continue
            taken[clip.set] = taken.get(clip.set, 0) + 1
            chosen.append(clip)
        return chosen


def fingerprint(clips: Iterable[Clip]) -> str:
    """What identifies a set of clips in the results: their ids, audio and references. Results compare only when
    their sets have the same fingerprint."""
    digest = hashlib.sha256()
    for clip in clips:
        reference = hashlib.sha256(clip.text.encode("utf-8")).hexdigest()
        digest.update(f"{clip.id} {clip.sha256} {reference}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


def load_manifest(path: pathlib.Path = MANIFEST) -> Manifest:
    """Read and check a manifest: every clip names a known set, its audio is stored under a relative path, and ids and
    paths are unique."""
    try:
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        sets = data["sets"]
        clips = tuple(Clip(**clip) for clip in data["clips"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise DataError(f"can't read the manifest {path}: {e}") from e
    for clip in clips:
        _check_clip(clip, sets)
    for field in ("id", "audio"):
        values = [getattr(clip, field) for clip in clips]
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise DataError(f"the manifest {path} repeats the {field} of {', '.join(duplicates)}")
    return Manifest(sets, clips)


def _check_clip(clip: Clip, sets: Dict[str, Any]) -> None:
    problem = ""
    audio = pathlib.PurePosixPath(clip.audio)
    if clip.set not in sets:
        problem = f"its set {clip.set!r} isn't in the manifest's sets"
    elif audio.is_absolute() or ".." in audio.parts or "\\" in clip.audio or ":" in clip.audio:
        problem = f"its audio path {clip.audio!r} isn't a relative path inside the cache folder"
    elif (clip.row is None) == (clip.member is None):
        problem = "it needs either a row (Parquet) or a member (tar archive)"
    elif len(clip.sha256) != 64 or set(clip.sha256) - set("0123456789abcdef"):
        problem = "its sha256 isn't 64 lower-case hex digits"
    elif not clip.text.strip() or not clip.seconds > 0:
        problem = "it has no reference text or no length"
    if problem:
        raise DataError(f"manifest clip {clip.id!r}: {problem}")


def default_data_dir() -> pathlib.Path:
    """BENCHMARK_DATA_DIR, or speech-toolkit-benchmarks in the user's cache folder (XDG_CACHE_HOME or ~/.cache)."""
    configured = os.getenv(DATA_DIR_VARIABLE)
    if configured:
        return pathlib.Path(configured).expanduser()
    cache = os.getenv("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache"
    return pathlib.Path(cache) / "speech-toolkit-benchmarks"


def sha256_of(path: pathlib.Path) -> Optional[str]:
    """The SHA-256 of the file at *path*, or None if there is no such file."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError:
        return None
    return digest.hexdigest()


def prepare(
    manifest: Manifest, data_dir: pathlib.Path, clips: Optional[Iterable[Clip]] = None, log: Log = log_to_stderr
) -> None:
    """Make sure every clip of *clips* (all of the manifest's by default) is in *data_dir* and intact: download the
    missing or damaged ones, and check each against its SHA-256.

    Raises:
        DataError: if a clip can't be downloaded, or what was downloaded doesn't match its SHA-256.
    """
    wanted = list(manifest.clips if clips is None else clips)
    missing = [clip for clip in wanted if sha256_of(data_dir / clip.audio) != clip.sha256]
    if not missing:
        log(f"The evaluation set is in {data_dir}: {len(wanted)} clips, all intact.")
        return
    groups: Dict[Tuple[str, str], List[Clip]] = {}
    for clip in missing:
        groups.setdefault((clip.set, clip.file), []).append(clip)
    for (name, file), group in groups.items():
        _download(manifest.sets[name], file, group, data_dir, log)
    log(f"The evaluation set is in {data_dir}: {len(wanted)} clips, all intact ({len(missing)} downloaded).")


def _download(source: Dict[str, Any], file: str, clips: List[Clip], data_dir: pathlib.Path, log: Log) -> None:
    """Download *clips* from *file* of *source*'s dataset, trying again after a network error."""
    where = f"{source['dataset']}@{source['revision'][:12]}: {file}"
    for attempt in range(1, ATTEMPTS + 1):
        todo = [clip for clip in clips if sha256_of(data_dir / clip.audio) != clip.sha256]
        if not todo:
            return
        log(f"Downloading {len(todo)} clip{'s' * (len(todo) != 1)} from {where}")
        try:
            with open_dataset_file(source["dataset"], source["revision"], file, streaming=todo[0].row is None) as f:
                found = read_rows(f, todo) if todo[0].row is not None else read_members(f, todo)
                for clip, data in found:
                    store(data_dir, clip, data)
            return
        except DataError:
            raise
        except Exception as e:  # network and server errors come in many types
            if attempt == ATTEMPTS:
                raise DataError(f"can't download {where}: {type(e).__name__}: {e}") from e
            log(f"Download failed ({type(e).__name__}: {e}); trying again")
            time.sleep(2**attempt)


def open_dataset_file(dataset: str, revision: str, file: str, streaming: bool) -> IO[bytes]:
    """Open *file* of the Hugging Face dataset *dataset* at *revision* for reading over HTTP: seekable (by ranges) for
    a Parquet file, or as one stream for an archive. Uses HF_TOKEN when it is set."""
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as e:
        raise DataError(
            f"downloading the evaluation set needs huggingface_hub (faster-whisper installs it): {e}"
        ) from e
    path = f"datasets/{dataset}@{revision}/{file}"
    opened: IO[bytes] = HfFileSystem().open(path, "rb", block_size=0 if streaming else None)
    return opened


def read_rows(parquet_file: IO[bytes], clips: Sequence[Clip], column: str = "audio") -> Iterator[Tuple[Clip, bytes]]:
    """The audio of *clips* from their rows of a Parquet file, whose *column* holds ``{"bytes": ..., "path": ...}``
    as Hugging Face datasets store audio. Only the row groups that hold them are read."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise DataError(f"reading the Parquet files of the evaluation set needs pyarrow: {INSTALL_HINT}") from e
    parquet = pq.ParquetFile(parquet_file)
    by_row: Dict[int, Clip] = {clip.row: clip for clip in clips if clip.row is not None}
    first = 0
    for group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        rows = sorted(row for row in by_row if first <= row < first + count)
        if rows:
            audio = parquet.read_row_group(group, columns=[column]).column(column)
            for row in rows:
                yield by_row.pop(row), audio[row - first]["bytes"].as_py()
        first += count
    if by_row:
        raise DataError(f"the Parquet file has {first} rows; no row {', '.join(map(str, sorted(by_row)))}")


def read_members(archive: IO[bytes], clips: Sequence[Clip]) -> Iterator[Tuple[Clip, bytes]]:
    """The audio of *clips* from their members of a (compressed) tar archive, read as a stream: reading stops once
    every clip is found."""
    by_member: Dict[str, Clip] = {clip.member: clip for clip in clips if clip.member is not None}
    with tarfile.open(fileobj=archive, mode="r|*") as tar:
        for member in tar:
            clip = by_member.pop(member.name, None)
            extracted = tar.extractfile(member) if clip is not None and member.isfile() else None
            if clip is not None and extracted is not None:
                yield clip, extracted.read()
            if not by_member:
                return
    raise DataError(f"the archive has no file named {', '.join(sorted(by_member))}")


def store(data_dir: pathlib.Path, clip: Clip, data: bytes) -> None:
    """Write a clip's audio to the cache folder, if it matches the manifest's SHA-256.

    Raises:
        DataError: if it doesn't match.
    """
    actual = hashlib.sha256(data).hexdigest()
    if actual != clip.sha256:
        raise DataError(
            f"{clip.id}: the downloaded audio has SHA-256 {actual}, but the manifest expects {clip.sha256}; "
            "the dataset changed, or the download is damaged"
        )
    path = data_dir / clip.audio
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, path)  # a reader never sees half a file


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.data", description="Download the benchmark's evaluation set and check it."
    )
    parser.add_argument(
        "--data-dir", type=pathlib.Path, help=f"where the clips are kept (default: {default_data_dir()})"
    )
    parser.add_argument("--manifest", type=pathlib.Path, default=MANIFEST, help="the manifest (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        prepare(load_manifest(args.manifest), args.data_dir or default_data_dir())
    except DataError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
