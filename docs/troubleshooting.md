# Troubleshooting

- **Out of memory:** use a smaller model (`-m small`) or `-b faster-whisper`.
- **Parakeet/Canary "requires extra packages":** `pip install "speech-transcription-toolkit[nvidia]"` (needs
  transformers 5.18+).
- **Diarization "could not download":**
  - Accept the terms of
    [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1).
  - Set `HF_TOKEN`; see [Speaker labels](diarization.md).
- **`ffmpeg` not found:** install it and make sure it is on your `PATH`.
- **"Failed to load audio: …":** ffmpeg couldn't decode the file; the message ends with ffmpeg's reason.
  - "the input has no audio stream": the file has no sound track.
  - "has its index at the end, so it can only be decoded from a path": an MP4, M4A or MOV file given as bytes or a
    file object. Pass its path instead, or move the index to the front: `ffmpeg -i in.m4a -c copy -movflags +faststart
    out.m4a`.
  - "isn't allowed in strict mode": `load_audio(strict=True)` takes only common audio and video formats; convert the
    file, to WAV or MP3 say.
  - "decoding took longer than the … s timeout": raise `load_audio()`'s `timeout`.
- **Made-up text in silent parts:** try `-b faster-whisper --vad`.
- **Translation keeps the original language:** Whisper's turbo models can't translate; use `-m medium` or `-m large-v3`.
- **Canary transcribes in the wrong language:** Canary can't detect the language; pass it with `-l`.
