# Benchmarks

One command measures backends and models on a fixed evaluation set: the word error rate (WER), the speed (RTFx) and
the peak memory. A weekly workflow runs the tiny Whisper models on the CPU and fails when their WER rises above the
committed baseline. The harness runs from a checkout and isn't part of the package. The numbers measured so far are
in [Backends](../docs/backends.md#benchmarks).

## Running it

```bash
pip install -e ".[whisper]" -r requirements-bench.txt   # plus the extras of the backends to measure
python -m benchmarks.run -b faster-whisper -m tiny
python -m benchmarks.run -b faster-whisper -m tiny -m base -b whisper -m tiny   # several models
python -m benchmarks.run -b parakeet                     # a backend's default model
python -m benchmarks.run -b faster-whisper -m tiny --sets librispeech --limit 10   # a quick look
```

- **Models:** each `-m` names a model of the `-b` before it; a `-b` without `-m` measures the backend's default model.
  Any registered backend works, so a new backend needs nothing here.
- **Output:** a Markdown table on stdout (progress goes to stderr), and the results as JSON in
  `benchmarks/results/<time>.json` (ignored by git), or where `--json` says.
- **Device:** `--device cpu` or `--device cuda`; by default each backend picks, as `transcribe` does.
- **Evaluation set:** the first run downloads it (about 40 MB of audio, out of some 370 MB read from Hugging Face) into
  `~/.cache/speech-toolkit-benchmarks`, or `$BENCHMARK_DATA_DIR`. `python -m benchmarks.data` downloads it without
  running anything.

### On a GPU

Install a CUDA build of torch first if the backend runs on torch (see
[GPU setup](../docs/install.md#gpu)), then:

```bash
python -m benchmarks.run --device cuda -b faster-whisper -m large-v3 -m turbo -b whisper -m turbo -b parakeet
```

The table then shows the GPU's name and its peak memory: torch's own count for the torch backends, and for
faster-whisper (marked `~`) the GPU's whole use above the level before the model loaded, sampled with `nvidia-smi`
every 0.2 s, since CTranslate2 keeps no count. Other programs using the GPU inflate that number.

## The evaluation set

| Set | Source | Clips | Audio | Languages | License |
|---|---|--:|--:|---|---|
| `librispeech` | [LibriSpeech](https://www.openslr.org/12) test-clean: read audiobooks, 40 speakers | 100 | 12.5 min | English | CC BY 4.0 |
| `fleurs` | [FLEURS](https://huggingface.co/datasets/google/fleurs) test: read Wikipedia sentences | 30 | 6.3 min | German, Spanish, French (10 each) | CC BY 4.0 |

[`manifest.json`](manifest.json) lists every clip: the dataset file on Hugging Face it comes from, at a fixed
revision (a row of LibriSpeech's Parquet file, a member of a FLEURS tar archive), the SHA-256 of its audio, its length
and its reference transcript. Only the clips are downloaded (the Parquet row groups that hold them, the start of each
archive), each is checked against its SHA-256 before it is stored and again before every run, and none goes in the
repository. [`make_manifest.py`](make_manifest.py) picked them:

- **LibriSpeech:** 100 of test-clean's 2620 utterances, evenly spaced in id order, so every speaker is in. Reference:
  the LibriSpeech transcript.
- **FLEURS:** the first 10 recordings of each language's test archive, which is sorted by file name, so that only the
  start of the archive is read. Reference: FLEURS's normalized transcription.

**Licenses and attribution.** Both datasets are under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The manifest redistributes their
reference transcripts, unchanged; the audio is downloaded from Hugging Face:

- LibriSpeech, by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, from LibriVox audiobooks:
  "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015. Hugging Face copy:
  [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr).
- FLEURS, by Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara
  Rivera and Ankur Bapna (Google): "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", IEEE
  SLT 2022. [`google/fleurs`](https://huggingface.co/datasets/google/fleurs).

To change the set, edit and run `python -m benchmarks.make_manifest`; then the baseline has to be measured again,
since results only compare on the same clips (see below).

## What is measured

Each model runs in a process of its own, so its peak memory is its own:

1. **Load:** the model loads, timed on its own. A first run's load includes the download.
2. **Warm-up:** the first clip is transcribed once, untimed and unscored.
3. **Each clip:** its audio is decoded (ffmpeg, 16 kHz mono) outside the timed region and passed to
   `Transcriber.transcribe()` as an array, with the clip's language. The random generators (Python's, NumPy's,
   torch's and CTranslate2's) are reset first, so a temperature fallback samples the same way on every run.

| Metric | Meaning |
|---|---|
| WER | word errors (substitutions, deletions, insertions) over the reference's words, over all of a set's clips |
| RTFx | seconds of audio per second of transcription: higher is faster; loading and decoding excluded |
| Load | seconds to load the model |
| Peak RSS | the measuring process's peak resident memory: VmHWM on Linux, `ru_maxrss` on macOS, the peak working set on Windows (with psutil) |
| Peak GPU | torch's peak reserved memory, or (`~`) the GPU's use sampled with `nvidia-smi` for CTranslate2 |

**Word error rate.** Reference and transcript are normalized with OpenAI Whisper's text normalizers, as Whisper's own
evaluation does (from the `whisper-normalizer` package): `EnglishTextNormalizer` for English, which also writes
numbers, currencies and British spellings one way ("ten" matches "10"), and `BasicTextNormalizer` for the other
languages (lower case, no punctuation or bracketed text). [jiwer](https://github.com/jitsi/jiwer) then aligns the
words. The integration tests' WER helper (`tests/integration/wer.py`) only lower-cases and strips punctuation: right
for its synthesized samples, but on LibriSpeech, whose references spell numbers out, it would count "10" against "ten"
as an error.

**Reproducibility.** Two runs on the same machine give the **same transcripts, so the same WER** (difference 0): the
decoding is deterministic (greedy or beam search, and seeded sampling when a temperature fallback kicks in). **RTFx**
agrees **within 5%** on an otherwise idle machine: two runs of faster-whisper `tiny` on a 4-CPU Xeon gave 6.63 and
6.62. Load times vary most (1.2 and 2.4 s in those two runs); read them as a rough guide.

Between machines, speed differs a lot and the WER a little. GitHub's `ubuntu-latest` runners are Intel or AMD
machines, and two runs of the workflow landed on one of each:

| | Intel Xeon Platinum 8370C | AMD EPYC 7763 |
|---|--:|--:|
| openai-whisper `tiny` (float32): WER LibriSpeech, FLEURS | 0.0778, 0.3244 | 0.0778, 0.3244 |
| openai-whisper `tiny`: RTFx | 18.3 | 11.4 |
| faster-whisper `tiny` (int8): WER LibriSpeech, FLEURS | 0.0720, 0.2837 | 0.0710, 0.3015 |
| faster-whisper `tiny`: RTFx | 12.7 | 12.8 |

CTranslate2's int8 kernels compute a little differently on the two CPUs, and the tiny model's transcripts of hard
clips (French above all) are fragile, so faster-whisper's FLEURS WER moved by 1.8 points; on a cloud VM with another
Intel Xeon it matched the Intel runner exactly. That is why FLEURS gets a wider tolerance than LibriSpeech (below), and
why the workflow doesn't gate on speed.

## The results file

`python -m benchmarks.run` writes one JSON document. Fields may be added; a field that changes meaning or goes bumps
`schema`, and `compare` refuses documents of another schema.

For example, from [`baseline.json`](baseline.json), shortened:

```jsonc
{
  "schema": 1,
  "created": "2026-10-27T12:57:16+00:00",
  "environment": {
    "machine": "Linux 7.0.0-1017-azure (x86_64)", "cpu": "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz",
    "cpus": 4, "ram_bytes": 16758403072, "python": "3.12.15", "speech_toolkit": "0.4.0",
    "packages": {"jiwer": "4.0.0", "whisper-normalizer": "0.1.15", "faster-whisper": "1.2.1", "...": "..."},
    "runner": "GitHub Actions (ubuntu26)", "commit": "7ef2f8f..."     // on GitHub Actions only, else null
  },
  "settings": {"device": "cpu", "isolated": true, "sets": ["fleurs", "librispeech"], "limit": null,
               "scoring": "whisper-normalizer 0.1.15, jiwer 4.0.0"},
  "results": [                      // one per backend and model, in the order measured
    {
      "backend": "faster-whisper", "model": "tiny",
      "device": "cpu", "precision": "int8_float32", "gpu": null,   // where the model actually ran
      "load_seconds": 1.861,
      "audio_seconds": 1132.225, "seconds": 88.983, "rtfx": 12.72,  // over all clips
      "peak_rss_bytes": 373780480, "peak_gpu_bytes": null, "peak_gpu_sampled": false,
      "sets": {
        "librispeech": {
          "clips": 100, "fingerprint": "0b0ec606cf74652e",  // the clips and references: compare only when equal
          "audio_seconds": 752.905, "seconds": 63.015, "rtfx": 11.95,
          "words": 2069, "errors": 149, "substitutions": 104, "deletions": 16, "insertions": 29, "wer": 0.072015,
          "languages": {"en": {"words": 2069, "errors": 149, "...": "...", "wer": 0.072015}}
        },
        "fleurs": {"...": "..."}
      },
      "clips": {                    // per clip, to see which transcripts changed
        "librispeech/1089-134686-0000": {"set": "librispeech", "audio_seconds": 10.435, "seconds": 0.7771,
                                         "words": 28, "errors": 2, "hypothesis": "He hoped there would be stew..."}
      }
    }
  ]
}
```

## Comparing two runs

```bash
python -m benchmarks.compare benchmarks/baseline.json benchmarks/results/20261027-123456.json
python -m benchmarks.compare before.json after.json --max-rtfx-drop 0.10   # also gate on speed, on your own machine
python -m benchmarks.compare before.json after.json --max-wer-increase 0.005 --max-wer-increase fleurs=0.01
```

It prints a Markdown table of each model's WER and RTFx on each set, before and after, and exits with 1 when:

- a set's WER rose by more than its tolerance, an absolute difference: **0.01 (one percentage point) for
  LibriSpeech**, about 21 of its 2069 words, and **0.03 for FLEURS**, about 24 of its 786 words, which move more from
  one CPU to another (see Reproducibility). `--max-wer-increase WER` changes the default, and `--max-wer-increase
  SET=WER` one set's. The clips whose errors rose most are listed;
- with `--max-rtfx-drop FRACTION`, a model's RTFx fell by more than that fraction (off by default);
- the two can't be compared: a model (backend, model and device) measured on one side only, a set measured on other
  clips or references (another fingerprint), or another `schema`.

Other versions of jiwer or whisper-normalizer, another precision and another machine are noted, not failed.

## The regression workflow

[`.github/workflows/benchmark.yml`](../.github/workflows/benchmark.yml) runs every Monday, by hand from the Actions
tab, and on pull requests that change `benchmarks/`, `requirements-bench.txt`, `tests/test_benchmarks.py` or the
workflow. On the runner's CPU it:

1. runs `tests/test_benchmarks.py` with the benchmark's packages installed;
2. restores the evaluation set and the models from the cache, and downloads and checks what's missing;
3. runs `python -m benchmarks.run --device cpu -b faster-whisper -m tiny -b whisper -m tiny`;
4. compares the results with [`baseline.json`](baseline.json): a WER more than its tolerance above it (one
   percentage point for LibriSpeech, three for FLEURS) fails the run. Speed and memory are only reported.

The tables go to the run's summary, and the results file to its `benchmark-results` artifact. A failed weekly run
opens an issue, "The weekly benchmark fails", or comments on it while it is open. The model libraries
are the newest releases `pyproject.toml` allows, so the weekly run catches upstream changes; jiwer and
whisper-normalizer are pinned in `requirements-bench.txt`.

**Updating the baseline**, after an intended change in accuracy, a new model in the workflow, or a new evaluation
set: download the `benchmark-results` artifact of a run of the workflow on the pull request, and commit it as
`benchmarks/baseline.json`. Measure it on GitHub's runner rather than your own machine, so that its speeds compare
with the workflow's.

**Adding a model to the workflow:** add `-m <model>` after its backend's `-b` (or `-b <backend> -m <model>`) in the
"Run the benchmark" step, and its install to the install step if it needs an extra; then update the baseline.
`tests/test_benchmarks.py` checks that the workflow and the baseline name the same models.
