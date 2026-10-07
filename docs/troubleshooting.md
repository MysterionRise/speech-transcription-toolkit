# Troubleshooting

- **Out of memory:** use a smaller model (`-m small`) or `-b faster-whisper`.
- **Parakeet/Canary "requires extra packages":** `pip install "speech-transcription-toolkit[nvidia]"` (needs
  transformers 5.18+).
- **Diarization "could not download":**
  - Accept the terms of
    [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1).
  - Set `HF_TOKEN`; see [Speaker labels](diarization.md).
- **`ffmpeg` not found:** install it and make sure it is on your `PATH`.
- **Made-up text in silent parts:** try `-b faster-whisper --vad`.
- **Translation keeps the original language:** Whisper's turbo models can't translate; use `-m medium` or `-m large-v3`.
- **Canary transcribes in the wrong language:** Canary can't detect the language; pass it with `-l`.
