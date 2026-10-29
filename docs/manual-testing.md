# Manual testing before a release

CI has no GPU. It runs small models (tiny, 0.6B, 1B) on CPU runners, with short synthesized speech. So before each
release, the maintainer runs a QA pass on real hardware:

- **Windows with an NVIDIA GPU:** the integration suite's full tier on the GPU, plus the checklist below.
- **An Apple Silicon Mac:** the integration suite on the CPU, plus the install checks.

The QA pass comes between the TestPyPI dry run and pushing the release tag (see
[Releasing](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/RELEASING.md)). Record it in a
[Release QA issue](https://github.com/MysterionRise/speech-transcription-toolkit/issues/new?template=release-qa.yml).
Fix anything it finds before tagging.

Plan for about two hours on Windows, most of it model downloads (about 20 GB the first time), and half an hour on the
Mac.

## The integration suite

`pytest -m integration` runs real models on the speech samples in `tests/data` (plain `pytest` skips them). These
environment variables control it:

| Variable | Values | What it does |
|---|---|---|
| `INTEGRATION_DEVICE` | `auto` (default), `cpu`, `cuda` | Passed as `--device` to every command and as `device=` to `Transcriber`. Each test checks the device its models actually ran on. With `cuda`, a test fails, saying why, when there's no GPU its backend can use; it never quietly runs on the CPU. `auto` lets each backend choose, as CI does. |
| `INTEGRATION_TIER` | `default`, `full` | `full` adds the larger models (see below). |
| `HF_TOKEN` | a Hugging Face token | Runs the speaker-label tests ([Speaker labels](diarization.md) says how to get access). |
| `INTEGRATION_REQUIRE_EXTRAS` | `1` | A missing package fails its tests instead of skipping them (CI sets it). |

**Default tier** (what CI runs): Whisper and faster-whisper `tiny`, Parakeet and Canary, on the two samples, plus the
long-audio test. The long-audio test joins the samples into a 104 s clip at test time. It checks that:
- the chunked backends cut the clip into several chunks;
- timestamps stay in order across the chunk boundaries;
- the word error rate stays within the backend's threshold;
- no backend warns that it cut text off.

**Full tier**, which the tests in `tests/integration/test_full_tier.py` add:
- faster-whisper `large-v3` and `turbo`, Whisper `turbo` and `large-v3`, Parakeet v2 and v3, Canary and Voxtral Mini,
  each on the long clip;
- with `HF_TOKEN`, speaker labels on the long clip with faster-whisper `turbo` and pyannote in one process, which
  shares the GPU with `INTEGRATION_DEVICE=cuda`.

A model that can't fit is skipped with the reason. `MEMORY_GB` in `tests/integration/devices.py` holds approximate
needs, and a model may use 85% of the GPU's memory (or the RAM, on the CPU):

| Model | GPU (GB) | CPU RAM (GB) |
|---|--:|--:|
| faster-whisper `large-v3` | 5 | 4 |
| faster-whisper `turbo` | 3 | 3 |
| Whisper `turbo` | 6 | 6 |
| Whisper `large-v3` | 10 | 10 |
| Parakeet v2, v3 | 3 | 4 |
| Canary | 4 | 6 |
| Voxtral Mini | 11 | 20 |
| pyannote (speaker labels) | 1 | 2 |

So on a GPU with less than 12 GB, Voxtral Mini and Whisper `large-v3` are skipped. If a model that was allowed runs
out of memory, or one that was skipped would fit, correct its line from the report's peak GPU column.

**The report:** at the end, the suite writes `integration-report.md` to pytest's base temp folder (`--basetemp`) and
prints it. It has:
- the machine, the settings and the package versions;
- one row per backend, model and device: the files transcribed, audio length, model load time, wall time, real-time
  factor (wall time / audio length), word error rate and peak GPU memory;
- the skipped tests with their reasons, and the failed tests.

The peak GPU memory is torch's own count for the torch backends. For faster-whisper it is marked `~`: CTranslate2
keeps no count, so the suite samples the whole GPU's use with `nvidia-smi` and reports it above the level before the
model loaded. Other programs using the GPU at the same time inflate it.

The suite tests the checkout's code with the packages installed in the virtual environment, so check out the commit
you are releasing. The installed wheel itself is checked by the clean installs and your own recordings below.

## 1. Setup

### Windows with an NVIDIA GPU

1. Install the NVIDIA driver; `nvidia-smi` shows the GPU. Install 64-bit Python 3.12 or 3.13 from python.org, and
   Git.
2. Install ffmpeg, then open a new terminal so that it's on the `PATH`:
   ```powershell
   winget install --id Gyan.FFmpeg
   ffmpeg -version
   ```
3. Get the checkout at the release commit, a new virtual environment, and the release's wheel from TestPyPI (the dry
   run uploaded it; its dependencies come from PyPI when you install it):
   ```powershell
   git clone https://github.com/MysterionRise/speech-transcription-toolkit.git
   cd speech-transcription-toolkit
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -U pip
   pip download --no-deps --only-binary :all: --index-url https://test.pypi.org/simple/ --dest dist `
       speech-transcription-toolkit==0.4.0
   $wheel = (Get-ChildItem dist\*.whl).FullName
   ```
   If PowerShell refuses to run `Activate.ps1`, allow local scripts once with
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
4. Optional: Hugging Face caches models with symlinks, which Windows allows only in Developer Mode. Without it,
   models are copied instead, with a warning; `$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"` hides the warning.

### Apple Silicon Mac

```bash
brew install ffmpeg python@3.12
git clone https://github.com/MysterionRise/speech-transcription-toolkit.git
cd speech-transcription-toolkit
python3.12 -m venv .venv && . .venv/bin/activate
pip download --no-deps --only-binary :all: --index-url https://test.pypi.org/simple/ --dest dist \
    speech-transcription-toolkit==0.4.0
wheel=$(ls dist/*.whl)
```

## 2. Clean installs from the TestPyPI build

On both machines, in the new virtual environment (`$wheel` is the downloaded wheel):

1. **The core install imports no torch:**
   ```bash
   pip install "$wheel"
   python -c "import sys, speech_toolkit.cli; print('torch' in sys.modules)"   # False
   pip show torch                                 # not found
   transcribe --version
   transcribe --list-backends
   transcribe tests/data/speech.flac -m tiny      # prints "Hello world. This is a speech recognition test."
   ```
2. **Windows only: torch with CUDA**, before any extra that needs torch. PyPI's torch for Windows runs on the CPU only,
   so it comes from PyTorch's index; pick the CUDA version on [pytorch.org](https://pytorch.org/get-started/locally/):
   ```powershell
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   pip keeps this torch when it installs the extras. After the last install, `torch.__version__` must still end in
   `+cu128`.
3. **Each extra in turn:** `pip install "${wheel}[<extra>]"` for every extra in `pyproject.toml`, `all` last. After each
   one, `transcribe --list-backends` shows its backend as installed (`transcribe-server --help` works after `server`).
   The suite runs every model next.

## 3. The integration suite

### Windows, on the GPU, full tier

1. CTranslate2 (faster-whisper) needs cuBLAS for CUDA 12, and before CTranslate2 4.6.3 cuDNN 9, which it doesn't
   bundle ([GPU setup](install.md#gpu) has the details). Install them with pip, and put their folders on the `PATH` in
   every new terminal (or add them to your user `PATH`):
   ```powershell
   pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
   $sp = python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
   $env:PATH = "$sp\nvidia\cublas\bin;$sp\nvidia\cudnn\bin;$env:PATH"
   transcribe tests\data\speech.flac -b faster-whisper -m tiny --device cuda   # prints the sentence
   ```
2. Install the suite's tools, and run it from the checkout, with the virtual environment active:
   ```powershell
   pip install -r requirements-dev.txt openai
   $env:INTEGRATION_DEVICE = "cuda"
   $env:INTEGRATION_TIER = "full"
   $env:HF_TOKEN = "hf_..."
   python -m pytest -m integration --no-cov -rA --basetemp "$env:TEMP\stt-qa"
   ```

### Mac, on the CPU

```bash
pip install -r requirements-dev.txt openai
INTEGRATION_DEVICE=cpu HF_TOKEN=hf_... python -m pytest -m integration --no-cov -rA --basetemp /tmp/stt-qa
```

### Results

- Everything passes. On Windows, the models that don't fit in the GPU are skipped (see the table above).
- `--basetemp` names the folder for the test outputs (transcripts, subtitles, the server's log, each command's
  `instrument-events.jsonl`) and `integration-report.md`. pytest empties it at the start of each run, so keep the
  report for the Release QA issue.
- The full tier also runs on the CPU (`INTEGRATION_DEVICE=cpu INTEGRATION_TIER=full`), slowly; Voxtral needs about 20
  GB of RAM there.

## 4. Your own recordings

With the installed `transcribe` command, on Windows (on the GPU) unless a step says otherwise:

- [ ] **A 10-minute podcast:** `transcribe podcast.mp3 -b faster-whisper -o podcast.srt`. Open the subtitles in a
  player (VLC or mpv). They stay in sync to the end, and there's no made-up text in pauses or at the end.
- [ ] **A meeting with 2 or 3 speakers:** `transcribe meeting.m4a --diarize -o meeting.txt`. The speaker labels follow
  the voices.
- [ ] **A clip in another language:**
  - `transcribe clip.mp3` detects the language and transcribes it;
  - `transcribe clip.mp3 -m medium --task translate` translates it to English;
  - `transcribe clip.mp3 -b canary -l de --task translate` (with the clip's language) translates it with Canary.
- [ ] **A video file:** `transcribe talk.mp4 -o talk.vtt`. The subtitles play in a player.
- [ ] **A Windows path with a drive letter and spaces:**
  `transcribe "C:\Users\me\My Recordings\talk.m4a" -o "C:\Users\me\My Recordings\talk.srt"`. ffmpeg gets paths with a
  `file:` prefix (#18), so this checks that a drive letter isn't taken for a protocol.
- [ ] **An M4A phone recording**, on both machines: an M4A can only be decoded from a path, not a pipe, so this checks
  that the file is handed to ffmpeg by its path.
- [ ] **Devices and precision** (#29):
  - on Windows, `transcribe talk.mp3 --device cuda:0 --compute-type auto` runs on the GPU;
  - on the Mac, `transcribe talk.mp3 -b parakeet --device mps` runs on the Apple GPU, and
    `transcribe talk.mp3 --device mps` (faster-whisper) stops at once, saying faster-whisper has no `mps`.
- [ ] **A folder, resumed** (#32): `transcribe recordings -f srt,json --outdir subs --skip-existing`, Ctrl-C during
  the second file, then the same command again. The second run skips the finished files and writes the rest.
- [ ] **The Docker images**, if Docker is installed (#35): `docker run --rm -v "$PWD:/data"
  ghcr.io/mysterionrise/speech-transcription-toolkit transcribe /data/talk.mp3` prints the transcript, and on Windows
  with the NVIDIA Container Toolkit, the `:cuda` image with `--gpus all` runs on the GPU.

## 5. Error paths

Each one fails cleanly: a message that says what went wrong, and a non-zero exit code.

- [ ] **The GPU without the CUDA libraries** (#56): in a new terminal, without the cuBLAS and cuDNN folders on the
  `PATH` or under `%CUDA_PATH%`, run `transcribe talk.mp3 -b faster-whisper`. It prints one warning naming the missing
  library and transcribes on the CPU. With `--device cuda` added, it stops with the same message instead. Neither
  crashes or hangs.
- [ ] **A model too large for the GPU:** `transcribe talk.mp3 -b voxtral --device cuda` on a GPU with less than 12 GB.
  It reports the out-of-memory error.
- [ ] **Ctrl-C during a batch:** `transcribe recordings --outdir subs`, then Ctrl-C during the second file. It stops
  within a few seconds; the finished files are complete, and there's no half-written file.
- [ ] **A corrupt file in a batch:** add a text file renamed to `broken.mp3` to the folder. The batch reports
  `Error: ...broken.mp3: ...`, transcribes the other files, and ends with `Error: 1 of N files failed.`

## 6. The server, from a second machine

Start the server on the Windows machine with a key you make up; it reads the key from `TRANSCRIBE_API_KEY`. Allow
Python through the firewall on private networks when Windows asks:

```powershell
$env:TRANSCRIBE_API_KEY = "<a key you make up>"
transcribe-server -b faster-whisper -m turbo --host 0.0.0.0
```

From the Mac, with the same key in `KEY` (replace `windows-pc` with the Windows machine's address):

- [ ] **The API key works:**
  ```bash
  curl http://windows-pc:8000/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
      -F model=whisper-1 -F file=@talk.mp3
  ```
  The answer is the transcript, as JSON.
- [ ] **A wrong key gets 401 before the upload completes** (#21): send a large file (several hundred MB) with
  another key. The 401 comes back at once, not after the whole file has gone up.
- [ ] **An oversized upload gets 413:** send a file larger than `--max-upload-mb` (100 MB by default) with the right
  key.
- [ ] **`/docs` is hidden when a key is set:** `curl -i http://windows-pc:8000/docs` answers 404.

## 7. Record the pass

Open a [Release QA issue](https://github.com/MysterionRise/speech-transcription-toolkit/issues/new?template=release-qa.yml)
for the version. Fill in both machines, tick the steps that passed, and paste both `integration-report.md` files. Note
anything that failed, with a link to its issue. The release tag (step 5 of
[Releasing](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/RELEASING.md)) waits until the
issue shows both machines passing.
