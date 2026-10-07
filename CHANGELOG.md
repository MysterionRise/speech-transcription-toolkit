# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

Entries come from the fragments in [`changelog.d/`](https://github.com/MysterionRise/speech-transcription-toolkit/tree/main/changelog.d),
collected with towncrier when a release is made.

<!-- towncrier release notes start -->

## Before 0.4.0

Versions 0.1.0 to 0.3.0 were never published to PyPI. They added:

- **0.3.0:** `transcribe-server`, a local OpenAI-compatible speech-to-text API
  ([#11](https://github.com/MysterionRise/speech-transcription-toolkit/pull/11)).
- **0.2.0** ([#10](https://github.com/MysterionRise/speech-transcription-toolkit/pull/10)):
  - the `--prompt`, `--vad`, `--word-timestamps` and `--max-line-width` options;
  - word-level speaker labels;
  - the NVIDIA Parakeet and Canary backends.
- **0.1.0:** an installable package with a Python API (`Transcriber`, `transcribe`) and the `transcribe` and `ogg2wav`
  commands ([#8](https://github.com/MysterionRise/speech-transcription-toolkit/pull/8)).
- **Earlier work:**
  - pluggable backends, with Voxtral and faster-whisper;
  - SRT, WebVTT and JSON output;
  - batch mode;
  - speaker diarization with pyannote.audio
    ([#3](https://github.com/MysterionRise/speech-transcription-toolkit/pull/3)).
