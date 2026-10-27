"""Pick the clips of the evaluation set and write manifest.json (maintainers only: the benchmark only reads it).

    python -m benchmarks.make_manifest

- **LibriSpeech test-clean** (English read speech, 40 speakers): 100 of its 2620 utterances, evenly spaced in id order
  so that every speaker is in, from the Parquet file of ``openslr/librispeech_asr``.
- **FLEURS** (read Wikipedia sentences): the first 10 recordings of the test split in German, Spanish (Latin America)
  and French, as each language's archive in ``google/fleurs`` stores them (sorted by file name, which has nothing to
  do with the content), so that only the start of each archive is downloaded. The reference is FLEURS's normalized
  transcription.

Both datasets are pinned to a revision. Each clip is downloaded once to record its SHA-256 and length; the reference
transcripts go in the manifest, the audio never goes in the repository. Changing the set changes the fingerprint of
every result, so the baseline (baseline.json) has to be measured again.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import pathlib
import tarfile
from typing import Any, Dict, List

from benchmarks.data import MANIFEST, Clip, load_manifest, log_to_stderr, open_dataset_file, read_rows

LIBRISPEECH: Dict[str, Any] = {
    "title": "LibriSpeech test-clean",
    "dataset": "openslr/librispeech_asr",
    "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "license": "CC BY 4.0",
    "homepage": "https://www.openslr.org/12",
    "citation": (
        "V. Panayotov, G. Chen, D. Povey and S. Khudanpur, LibriSpeech: an ASR corpus based on public domain audio "
        "books, ICASSP 2015"
    ),
    "selection": "100 of the 2620 utterances of test-clean, evenly spaced in id order",
}
LIBRISPEECH_FILE = "all/test.clean/0000.parquet"
LIBRISPEECH_CLIPS = 100

FLEURS: Dict[str, Any] = {
    "title": "FLEURS test: German, Spanish (Latin America), French",
    "dataset": "google/fleurs",
    "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
    "license": "CC BY 4.0",
    "homepage": "https://huggingface.co/datasets/google/fleurs",
    "citation": (
        "A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa, C. Rivera and A. Bapna, FLEURS: "
        "Few-shot Learning Evaluation of Universal Representations of Speech, IEEE SLT 2022"
    ),
    "selection": (
        "the first 10 recordings of each language's test archive (sorted by file name); the reference is FLEURS's "
        "normalized transcription"
    ),
}
FLEURS_LANGUAGES = {"de": "de_de", "es": "es_419", "fr": "fr_fr"}  # ISO 639-1 code: FLEURS's name
FLEURS_CLIPS = 10  # per language


def described(clip: Clip, data: bytes) -> Clip:
    """*clip*, with where its audio *data* goes in the cache folder, and the data's SHA-256 and length."""
    from speech_toolkit.media import SAMPLE_RATE, load_audio

    extension = ".flac" if data[:4] == b"fLaC" else ".wav"
    seconds = round(len(load_audio(data)) / SAMPLE_RATE, 3)
    return dataclasses.replace(
        clip, audio=clip.id + extension, sha256=hashlib.sha256(data).hexdigest(), seconds=seconds
    )


def librispeech_clips() -> List[Clip]:
    """LIBRISPEECH_CLIPS utterances of test-clean, evenly spaced in id order."""
    import pyarrow.parquet as pq

    source = LIBRISPEECH
    with open_dataset_file(source["dataset"], source["revision"], LIBRISPEECH_FILE, streaming=False) as file:
        rows = pq.ParquetFile(file).read(columns=["id", "text"]).to_pylist()
        number = {row["id"]: index for index, row in enumerate(rows)}
        ids = sorted(number)
        chosen = [ids[i * len(ids) // LIBRISPEECH_CLIPS] for i in range(LIBRISPEECH_CLIPS)]
        wanted = [
            Clip(f"librispeech/{utterance}", "librispeech", "en", LIBRISPEECH_FILE, "", "", 0.0, text, row=row)
            for utterance, row, text in ((u, number[u], rows[number[u]]["text"]) for u in chosen)
        ]
        clips = [described(clip, data) for clip, data in read_rows(file, wanted)]
    return sorted(clips, key=lambda clip: clip.id)


def fleurs_transcripts(name: str) -> Dict[str, str]:
    """The normalized transcription of each recording of FLEURS's test split in *name* (such as fr_fr), by file."""
    source = FLEURS
    with open_dataset_file(source["dataset"], source["revision"], f"data/{name}/test.tsv", streaming=True) as file:
        text = file.read().decode("utf-8")
    # Columns: id, file name, raw transcription, transcription, phonemes, number of samples, gender.
    rows = csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
    return {row[1]: row[3] for row in rows if len(row) > 3}


def fleurs_clips(language: str, name: str) -> List[Clip]:
    """The first FLEURS_CLIPS recordings of the test archive of *name*, with their transcripts."""
    transcripts = fleurs_transcripts(name)
    file = f"data/{name}/audio/test.tar.gz"
    clips: List[Clip] = []
    source = FLEURS
    with open_dataset_file(source["dataset"], source["revision"], file, streaming=True) as archive:
        with tarfile.open(fileobj=archive, mode="r|*") as tar:
            for member in tar:
                path = pathlib.PurePosixPath(member.name)
                extracted = tar.extractfile(member) if member.isfile() and path.name in transcripts else None
                if extracted is None:
                    continue
                clip_id = f"fleurs/{language}/{path.stem}"
                clip = Clip(clip_id, "fleurs", language, file, "", "", 0.0, transcripts[path.name], member=member.name)
                clips.append(described(clip, extracted.read()))
                if len(clips) == FLEURS_CLIPS:
                    break
    return clips


def main() -> None:
    log_to_stderr(f"Picking {LIBRISPEECH_CLIPS} LibriSpeech utterances")
    clips = librispeech_clips()
    for language, name in FLEURS_LANGUAGES.items():
        log_to_stderr(f"Picking {FLEURS_CLIPS} FLEURS recordings in {name}")
        clips += fleurs_clips(language, name)
    manifest = {
        "description": (
            "The benchmark's evaluation set (see benchmarks/README.md), made by python -m benchmarks.make_manifest. "
            "python -m benchmarks.data downloads the audio and checks each clip against its sha256."
        ),
        "sets": {"librispeech": LIBRISPEECH, "fleurs": FLEURS},
        "clips": [{key: value for key, value in vars(clip).items() if value is not None} for clip in clips],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    load_manifest(MANIFEST)  # check it
    minutes = sum(clip.seconds for clip in clips) / 60
    log_to_stderr(f"Wrote {MANIFEST}: {len(clips)} clips, {minutes:.1f} minutes of audio")


if __name__ == "__main__":
    main()
