# Voice activity detection

What `--vad` (`vad=True`) runs: Silero VAD, the model faster-whisper ships, finds the stretches of speech in decoded
audio, so that the backends skip the silence. It is in
[speech_toolkit/vad.py](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/speech_toolkit/vad.py);
[Voice activity detection](../backends.md#voice-activity-detection) describes what each backend does with it, and its
options.

::: speech_toolkit.vad.detect_speech

::: speech_toolkit.vad.has_speech

::: speech_toolkit.vad.SpeechRegion

::: speech_toolkit.vad.SpeechMap

::: speech_toolkit.media.split_audio

::: speech_toolkit.vad.VAD_OPTIONS
