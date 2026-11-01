"""Inference options (#36): faster-whisper's batched pipeline, beam size, CPU threads and workers; Whisper's controls
against made-up text, for openai-whisper and faster-whisper; batch sizes for Parakeet, Canary and Voxtral; the
--batch-size and --beam-size flags; and backend options in the benchmark harness.

faster_whisper, ctranslate2, whisper, torch and transformers are mocks in sys.modules: each option is checked where it
reaches the library's call.
"""

from __future__ import annotations

import math
import pathlib
import sys
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit import (
    ProgressEvent,
    SpeechToolkitWarning,
    Transcriber,
    TranscriptionBackend,
    TranscriptionResult,
    UnsupportedOptionError,
    cli,
)
from speech_toolkit.backends import BackendOption, FasterWhisperBackend, ParakeetBackend, faster_whisper_backend

RATE = 16000


@pytest.fixture(autouse=True)
def no_color(monkeypatch):
    """Python 3.14's argparse can colour help and errors; these tests match the plain text."""
    monkeypatch.setenv("PYTHON_COLORS", "0")


@pytest.fixture
def audio(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "talk.wav"
    path.touch()
    return path


###############################################################################
# faster-whisper
###############################################################################


def word(text: str, start: float, end: float) -> SimpleNamespace:
    """A faster-whisper Word."""
    return SimpleNamespace(word=text, start=start, end=end, probability=0.9)


def segment(number: int, start: float, end: float, text: str, words: Any = None) -> SimpleNamespace:
    """A faster-whisper Segment, numbered from 1 as faster-whisper does."""
    return SimpleNamespace(
        id=number,
        start=start,
        end=end,
        text=text,
        tokens=[1],
        temperature=0.0,
        avg_logprob=-0.1,
        compression_ratio=1.0,
        no_speech_prob=0.0,
        words=words,
    )


# What the mocked model and its batched pipeline hear: two sentences, with word timings (faster-whisper leaves words
# None unless asked for them; the mocks always have them, so the tests see which ones the backend keeps).
SEGMENTS = [
    segment(1, 0.0, 2.0, " One two.", [word(" One", 0.0, 0.8), word(" two.", 0.9, 2.0)]),
    segment(2, 2.5, 4.0, " Three.", [word(" Three.", 2.6, 4.0)]),
]


@pytest.fixture
def fw(monkeypatch):
    """Mocked faster_whisper and ctranslate2, on a machine without a GPU: the model and its batched pipeline both hear
    SEGMENTS in 10 s of audio."""
    module, ctranslate2 = MagicMock(), MagicMock()
    ctranslate2.__version__ = "4.8.3"
    ctranslate2.get_cuda_device_count.return_value = 0
    ctranslate2.get_supported_compute_types.return_value = {"int8", "float32"}
    monkeypatch.setattr(faster_whisper_backend, "_cuda_library_problem", lambda version: None)
    info = SimpleNamespace(language="en", duration=10.0, language_probability=0.9)
    model, pipeline = module.WhisperModel.return_value, module.BatchedInferencePipeline.return_value
    model.transcribe.side_effect = lambda audio, **kwargs: (iter(SEGMENTS), info)
    pipeline.transcribe.side_effect = lambda audio, **kwargs: (iter(SEGMENTS), info)
    with patch.dict(sys.modules, {"faster_whisper": module, "ctranslate2": ctranslate2}):
        yield SimpleNamespace(module=module, model=model, pipeline=pipeline)


# What faster-whisper's transcribe() got before the options existed, and still gets with their defaults.
USUAL = {
    "language": None,
    "task": "transcribe",
    "log_progress": False,
    "initial_prompt": None,
    "vad_filter": False,
    "word_timestamps": False,
}


class TestFasterWhisper:
    def test_defaults_change_nothing(self, fw, audio):
        Transcriber("faster-whisper", "tiny").transcribe(audio, verbose=False)

        fw.module.WhisperModel.assert_called_once_with("tiny", device="cpu", compute_type="int8")
        fw.model.transcribe.assert_called_once_with(str(audio), **USUAL)
        fw.module.BatchedInferencePipeline.assert_not_called()

    @pytest.mark.parametrize(
        "options, passed",
        [
            ({"beam_size": "1"}, {"beam_size": 1}),
            ({"beam_size": 8}, {"beam_size": 8}),
            ({"condition_on_previous_text": "false"}, {"condition_on_previous_text": False}),
            ({"no_speech_threshold": "0.3"}, {"no_speech_threshold": 0.3}),
            # It needs word timings, which are computed for it.
            (
                {"hallucination_silence_threshold": "2"},
                {"hallucination_silence_threshold": 2.0, "word_timestamps": True},
            ),
        ],
    )
    def test_decoding_options_reach_transcribe(self, fw, audio, options, passed):
        Transcriber("faster-whisper", "tiny", options=options).transcribe(audio, verbose=False)

        fw.model.transcribe.assert_called_once_with(str(audio), **{**USUAL, **passed})

    def test_options_at_their_defaults_are_left_out(self, fw, audio):
        options = {"beam_size": 5, "condition_on_previous_text": True, "no_speech_threshold": 0.6, "batch_size": 1}
        Transcriber("faster-whisper", "tiny", options={**options, "hallucination_silence_threshold": 0})

        Transcriber("faster-whisper", "tiny", options=options).transcribe(audio, verbose=False)

        assert fw.model.transcribe.call_args.kwargs == USUAL

    def test_threads_and_workers_reach_the_model(self, fw):
        Transcriber("faster-whisper", "tiny", options={"cpu_threads": "2", "num_workers": "3"})

        fw.module.WhisperModel.assert_called_once_with(
            "tiny", device="cpu", compute_type="int8", cpu_threads=2, num_workers=3
        )

    @pytest.mark.parametrize("asked, kept", [(False, False), (True, True)])
    def test_word_timings_computed_for_the_silence_threshold_are_kept_only_when_asked(self, fw, audio, asked, kept):
        transcriber = Transcriber("faster-whisper", "tiny", options={"hallucination_silence_threshold": 1.5})

        result = transcriber.transcribe(audio, word_timestamps=asked, verbose=False)

        assert fw.model.transcribe.call_args.kwargs["word_timestamps"] is True
        assert all(("words" in seg) is kept for seg in result.segments)

    def test_batch_size_runs_the_batched_pipeline(self, fw, audio):
        transcriber = Transcriber("faster-whisper", "tiny", options={"batch_size": "8", "beam_size": "2"})

        transcriber.transcribe(audio, language="de", verbose=False)

        fw.module.BatchedInferencePipeline.assert_called_once_with(fw.model)
        # The pipeline cuts the audio where voice activity detection finds speech, also without --vad. It isn't asked
        # for timestamps (without_timestamps=False), with which it loses text.
        fw.pipeline.transcribe.assert_called_once_with(
            str(audio), **{**USUAL, "language": "de", "vad_filter": True}, batch_size=8, beam_size=2
        )
        fw.model.transcribe.assert_not_called()

    def test_the_batched_pipeline_keeps_word_timestamps_and_segment_order(self, fw, audio):
        events: List[ProgressEvent] = []
        transcriber = Transcriber("faster-whisper", "tiny", options={"batch_size": 4})

        result = transcriber.transcribe(audio, word_timestamps=True, verbose=False, progress=events.append)

        assert fw.pipeline.transcribe.call_args.kwargs["word_timestamps"] is True
        assert [(s["id"], s["start"], s["end"], s["text"]) for s in result.segments] == [
            (0, 0.0, 2.0, " One two."),
            (1, 2.5, 4.0, " Three."),
        ]
        assert result.segments[0]["words"] == [
            {"word": " One", "start": 0.0, "end": 0.8, "probability": 0.9},
            {"word": " two.", "start": 0.9, "end": 2.0, "probability": 0.9},
        ]
        assert result.text == "One two. Three."
        # Each segment is reported as the pipeline gives it, in order.
        assert [event.segment for event in events if event.segment is not None] == result.segments

    def test_a_new_pipeline_for_each_transcription(self, fw, audio):
        """The pipeline keeps the last word's time between batches; a transcription stopped halfway would leave it."""
        transcriber = Transcriber("faster-whisper", "tiny", options={"batch_size": 2})

        transcriber.transcribe(audio, verbose=False)
        transcriber.transcribe(audio, verbose=False)

        assert fw.module.BatchedInferencePipeline.call_count == 2

    def test_the_batched_pipeline_ignores_the_sequential_options_with_a_warning(self, fw, audio):
        options = {"batch_size": 4, "condition_on_previous_text": False, "hallucination_silence_threshold": 1.0}

        with pytest.warns(SpeechToolkitWarning) as warned:
            transcriber = Transcriber("faster-whisper", "tiny", options={**options, "beam_size": 1})
        transcriber.transcribe(audio, verbose=False)

        assert [str(warning.message) for warning in warned] == [
            "faster-whisper's batched pipeline (batch_size above 1) ignores condition_on_previous_text and "
            "hallucination_silence_threshold; it skips silence with voice activity detection instead."
        ]
        kwargs = fw.pipeline.transcribe.call_args.kwargs
        assert {"condition_on_previous_text", "hallucination_silence_threshold"}.isdisjoint(kwargs)
        assert (kwargs["beam_size"], kwargs["word_timestamps"]) == (1, False)

    def test_listed_with_their_defaults(self, capsys):
        cli.show_backends()
        lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
        listed = lines[lines.index("faster-whisper (default)") :]

        assert (
            "-O num_workers=INT: Transcriptions the model runs at once when several threads share it; each worker "
            "takes memory (default: 1)"
        ) in listed
        assert any(line.startswith("-O batch_size=INT: Pieces of audio decoded at a time;") for line in listed)
        assert "-O beam_size=INT: Beams in beam search; 1 decodes greedily, the fastest (default: 5)" in listed
        assert (
            "-O hallucination_silence_threshold=NUMBER: Skip silences longer than this many seconds around text that "
            "looks made up; turns on word timings (0: off) (default: 0.0)"
        ) in listed


###############################################################################
# openai-whisper
###############################################################################


@pytest.fixture
def whisper():
    """Mocked openai-whisper on the CPU, whose model hears " Hi" with its word timing."""
    module = MagicMock()
    model = module.load_model.return_value
    model.parameters.side_effect = lambda: iter([SimpleNamespace(device="cpu")])
    hi = {"word": " Hi", "start": 0.0, "end": 0.4, "probability": 0.9}
    model.transcribe.return_value = {
        "text": " Hi",
        "segments": [{"id": 0, "start": 0.0, "end": 0.4, "text": " Hi", "words": [hi]}],
        "language": "en",
    }
    with patch.dict(sys.modules, {"whisper": module}):
        yield model


class TestWhisper:
    def test_defaults_change_nothing(self, whisper, audio):
        """openai-whisper decodes greedily by default, without beam_size."""
        Transcriber("whisper", "tiny").transcribe(audio, verbose=False)

        whisper.transcribe.assert_called_once_with(str(audio), task="transcribe", verbose=None, fp16=False)

    @pytest.mark.parametrize(
        "options, passed",
        [
            ({"beam_size": "5"}, {"beam_size": 5}),
            ({"beam_size": 1}, {}),  # greedy, as without it
            ({"condition_on_previous_text": False}, {"condition_on_previous_text": False}),
            ({"no_speech_threshold": 0.45}, {"no_speech_threshold": 0.45}),
            ({"hallucination_silence_threshold": 2}, {"hallucination_silence_threshold": 2.0, "word_timestamps": True}),
        ],
    )
    def test_options_reach_transcribe(self, whisper, audio, options, passed):
        Transcriber("whisper", "tiny", options=options).transcribe(audio, verbose=False)

        whisper.transcribe.assert_called_once_with(str(audio), task="transcribe", verbose=None, fp16=False, **passed)

    @pytest.mark.parametrize("asked", [False, True])
    def test_word_timings_computed_for_the_silence_threshold_are_kept_only_when_asked(self, whisper, audio, asked):
        transcriber = Transcriber("whisper", "tiny", options={"hallucination_silence_threshold": 1.0})

        result = transcriber.transcribe(audio, word_timestamps=asked, verbose=False)

        assert ("words" in result.segments[0]) is asked


###############################################################################
# Parakeet, Canary and Voxtral
###############################################################################


class FakeBatch(dict):
    """A processor's output; for Parakeet, ``input_features`` stands for each chunk by its number of samples."""

    def __init__(self, rows: int = 1, **items: Any) -> None:
        super().__init__(**items)
        self.input_ids = SimpleNamespace(shape=(rows, 0))  # Voxtral's prompt: no tokens

    def to(self, *args: Any, **kwargs: Any) -> "FakeBatch":
        return self


@pytest.fixture
def transformers(monkeypatch):
    """Mocked torch and transformers, without a GPU. Any audio file decodes to ``state.seconds`` of silence.

    Parakeet hears one word per chunk at 1 s into it, the chunk's number of samples (" 1840800."); Canary and Voxtral
    "Hi." in each chunk.
    """
    torch, transformers = MagicMock(), MagicMock()
    torch.cuda.is_available.return_value = False
    state = SimpleNamespace(seconds=370)
    for module in ("nvidia_backend", "voxtral_backend"):
        monkeypatch.setattr(
            f"speech_toolkit.backends.{module}.load_audio", lambda path: np.zeros(state.seconds * RATE, np.float32)
        )
    processor = transformers.AutoProcessor.from_pretrained.return_value
    processor.side_effect = lambda audio, **kwargs: FakeBatch(input_features=[len(chunk) for chunk in audio])
    processor.decode.side_effect = lambda sequences, durations: (
        "",
        [[{"token": f" {samples}.", "start": 1.0, "end": 1.5}] for samples in sequences],
    )
    parakeet = transformers.ParakeetForTDT.from_pretrained.return_value.to.return_value
    parakeet.generate.side_effect = lambda input_features, **kwargs: SimpleNamespace(
        sequences=input_features, durations=None
    )
    processor.apply_transcription_request.side_effect = lambda audio, **kwargs: FakeBatch(len(audio), chunks=len(audio))
    processor.batch_decode.side_effect = lambda generated, **kwargs: ["Hi."] * len(generated)
    for name in ("CanaryForConditionalGeneration", "VoxtralForConditionalGeneration"):
        model = getattr(transformers, name).from_pretrained.return_value.to.return_value
        model.generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=0)
        model.generate.side_effect = lambda chunks=1, **kwargs: np.full((chunks, 2), 5)
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        yield SimpleNamespace(state=state, processor=processor, parakeet=parakeet)


# 370 s of silence in chunks of up to 120 s, each cut 5 s before the limit (plus half a 100 ms window).
PARAKEET_CHUNKS = [(0.0, 115.05), (115.05, 230.1), (230.1, 345.15), (345.15, 370.0)]


class TestParakeet:
    @pytest.mark.parametrize(
        "options, batches", [({}, [2, 2]), ({"batch_size": 3}, [3, 1]), ({"batch_size": 1}, [1] * 4)]
    )
    def test_chunks_are_transcribed_batch_size_at_a_time(self, transformers, audio, options, batches):
        result = Transcriber("parakeet", options=options).transcribe(audio, word_timestamps=True, verbose=False)

        calls = transformers.processor.call_args_list
        assert [len(call.args[0]) for call in calls] == batches
        assert all(call.kwargs == {"sampling_rate": RATE} for call in calls)
        # Each chunk's words, in order, at its own offset: one segment per chunk here (the pauses between them).
        sizes = [round((end - start) * RATE) for start, end in PARAKEET_CHUNKS]
        assert [(seg["start"], seg["text"]) for seg in result.segments] == [
            (round(start + 1.0, 3), f" {samples}.") for (start, _), samples in zip(PARAKEET_CHUNKS, sizes)
        ]
        assert result.segments[1]["words"] == [{"word": f" {sizes[1]}.", "start": 116.05, "end": 116.55}]

    def test_generate_gets_parakeets_own_step_limit_for_the_longest_chunk(self, transformers, audio):
        """max_symbols_per_step for each encoder frame: one per 160 samples, 8 times fewer after subsampling."""
        transformers.parakeet.config = SimpleNamespace(
            max_symbols_per_step=10, encoder_config=SimpleNamespace(subsampling_factor=8)
        )
        transformers.processor.feature_extractor = SimpleNamespace(hop_length=160)

        Transcriber("parakeet", options={"batch_size": 3}).transcribe(audio, verbose=False)

        longest, last = round(115.05 * RATE), round(24.85 * RATE)
        assert [call.kwargs["max_new_tokens"] for call in transformers.parakeet.generate.call_args_list] == [
            10 * math.ceil((longest // 160 + 1) / 8),
            10 * math.ceil((last // 160 + 1) / 8),
        ]

    def test_the_step_limit_without_those_settings(self, transformers):
        backend = ParakeetBackend()
        backend.load_model("parakeet-tdt-0.6b-v3")
        transformers.parakeet.config = SimpleNamespace()
        transformers.processor.feature_extractor = None

        assert backend._max_steps(120 * RATE) == 10 * 1501  # 12001 feature frames, 1501 encoder frames

    def test_a_warning_shows_once_per_run_not_once_per_file(self, transformers, audio):
        """Hiding transformers' max_length warning in a catch_warnings() block made the others show again for every
        file: leaving the block resets Python's record of the warnings shown."""
        transformers.state.seconds = 5
        transcriber = Transcriber("parakeet")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")  # Python's default: each warning once per place
            for _ in range(3):
                transcriber.transcribe(audio, language="de", verbose=False)

        assert [str(warning.message) for warning in caught] == [
            "Parakeet detects the language itself; ignoring language 'de'."
        ]

    @pytest.mark.filterwarnings("default:Parakeet detects the language itself:speech_toolkit.SpeechToolkitWarning")
    def test_once_in_a_batch_of_files(self, transformers, tmp_path, capsys):
        transformers.state.seconds = 5
        files = [tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.wav"]
        for path in files:
            path.touch()

        cli.main([*map(str, files), "-b", "parakeet", "-l", "de", "-q", "--outdir", str(tmp_path / "out")])

        assert capsys.readouterr().err.count("Warning: Parakeet detects the language itself") == 1


@pytest.mark.parametrize("backend, language", [("canary", "en"), ("voxtral", None)])
class TestCanaryAndVoxtral:
    @pytest.mark.parametrize(
        "options, batches", [({}, [3]), ({"batch_size": "2"}, [2, 1]), ({"batch_size": 1}, [1] * 3)]
    )
    def test_batch_size(self, transformers, audio, backend, language, options, batches):
        """70 s: three chunks of up to 30 s, eight at a time by default."""
        transformers.state.seconds = 70

        result = Transcriber(backend, options=options).transcribe(audio, language, verbose=False)

        calls = transformers.processor.apply_transcription_request.call_args_list
        assert [len(call.kwargs["audio"]) for call in calls] == batches
        assert [seg["text"] for seg in result.segments] == ["Hi."] * 3

    def test_at_least_one(self, transformers, backend, language):
        with pytest.raises(UnsupportedOptionError, match="^option batch_size must be at least 1, not 0$"):
            Transcriber(backend, options={"batch_size": 0})


###############################################################################
# --batch-size and --beam-size
###############################################################################


class TestFlags:
    def test_they_reach_faster_whisper(self, fw, audio):
        cli.main([str(audio), "-q", "-m", "tiny", "--batch-size", "8", "--beam-size", "1"])

        kwargs = fw.pipeline.transcribe.call_args.kwargs
        assert (kwargs["batch_size"], kwargs["beam_size"]) == (8, 1)

    def test_they_win_over_options(self, fw, audio):
        args = cli.parse_args([str(audio), "-O", "beam_size=3", "--beam-size", "2", "-O", "batch-size=4"])

        transcriber = cli.load_transcriber(args)

        assert (transcriber.backend.options["beam_size"], transcriber.backend.options["batch_size"]) == (2, 4)

    def test_they_reach_parakeet(self, transformers, audio):
        cli.main([str(audio), "-q", "-b", "parakeet", "--batch-size", "4"])

        assert [len(call.args[0]) for call in transformers.processor.call_args_list] == [4]

    @pytest.mark.parametrize(
        "argv, message",
        [
            (
                ["-b", "whisper", "--batch-size", "4"],
                "the whisper backend takes no --batch-size; the backends that do: faster-whisper, voxtral, parakeet, "
                "canary",
            ),
            (
                ["-b", "parakeet", "--beam-size", "2"],
                "the parakeet backend takes no --beam-size; the backends that do: whisper, faster-whisper",
            ),
        ],
    )
    def test_a_backend_without_the_option_fails_before_loading(self, audio, capsys, argv, message):
        with pytest.raises(SystemExit) as error:
            cli.main([str(audio), *argv])

        assert error.value.code == f"Error: {message}"
        assert "Loading" not in capsys.readouterr().err

    @pytest.mark.parametrize("flag", ["--batch-size", "--beam-size"])
    def test_positive(self, capsys, flag):
        with pytest.raises(SystemExit):
            cli.parse_args(["talk.wav", flag, "0"])

        assert "must be a positive integer, got 0" in capsys.readouterr().err

    def test_help_names_the_backends(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "500")
        with pytest.raises(SystemExit):
            cli.parse_args(["--help"])
        group = capsys.readouterr().out.split("\ninference:\n", 1)[1].split("\n\n", 1)[0]

        assert "--batch-size N " in group
        assert "(faster-whisper, voxtral, parakeet, canary; the same as -O batch_size=N)." in group
        assert "--beam-size N " in group
        assert "(whisper, faster-whisper; the same as -O beam_size=N)." in group


###############################################################################
# The benchmark harness: -O per backend, kept in the results
###############################################################################


@pytest.fixture
def bench():
    """The benchmark harness, from a source checkout (the sdist has no benchmarks/)."""
    pytest.importorskip("benchmarks.run", reason="needs a source checkout: the sdist has no benchmarks/ directory")
    from benchmarks import compare, run

    return SimpleNamespace(run=run, compare=compare)


class BeamsBackend(TranscriptionBackend):
    """A backend with a beam_size option, which says the beam size it decoded with."""

    name = "beams"
    capabilities = frozenset({"waveform"})
    OPTIONS = {"beam_size": BackendOption(int, 5, "Beams", minimum=1)}

    @classmethod
    def available_models(cls):
        return ["small"]

    def load_model(self, model_name, device=None, *, options=None):
        self._options = self._resolve_options(options)
        self._model, self._device = object(), "cpu"

    def transcribe(self, audio, language=None, task="transcribe", verbose=True):
        return TranscriptionResult(text=f"beams {self._options['beam_size']}", segments=[])


def results_entry(options: Dict[str, Any], rtfx: float = 20.0, **extra: Any) -> Dict[str, Any]:
    """A model's results on one set, as python -m benchmarks.run writes them."""
    return {
        "backend": "faster-whisper",
        "model": "tiny",
        "device": "cpu",
        "precision": "int8",
        "rtfx": rtfx,
        "sets": {"librispeech": {"fingerprint": "aaaa", "wer": 0.07, "rtfx": rtfx, "clips": 1}},
        "clips": {},
        **({"options": options} if options is not None else {}),
        **extra,
    }


def results(*entries: Dict[str, Any]) -> Dict[str, Any]:
    from benchmarks import SCHEMA

    environment = {"machine": "Linux", "cpu": "Xeon", "cpus": 4}
    return {"schema": SCHEMA, "environment": environment, "settings": {}, "results": list(entries)}


class TestBenchmarkOptions:
    def test_each_option_belongs_to_the_backend_before_it(self, bench):
        argv = ["-b", "faster-whisper", "-m", "tiny", "-b", "faster-whisper", "-m", "tiny", "-O", "beam_size=1"]
        args = bench.run.parse_args([*argv, "-O", "batch-size=8", "-b", "whisper", "-m", "tiny"])

        assert args.specs == [
            ("faster-whisper", ["tiny"], {}),
            ("faster-whisper", ["tiny"], {"beam_size": "1", "batch_size": "8"}),
            ("whisper", ["tiny"], {}),
        ]
        assert bench.run.model_list(args.specs) == [
            ("faster-whisper", "tiny", {}),
            ("faster-whisper", "tiny", {"beam_size": 1, "batch_size": 8}),  # checked and converted
            ("whisper", "tiny", {}),
        ]

    @pytest.mark.parametrize(
        "argv, message",
        [
            (["-O", "beam_size=1"], "-O/--option sets an option of the -b/--backend before it: give -b first"),
            (["-b", "whisper", "-O", "beam_size"], "-O/--option takes KEY=VALUE, such as beam_size=1, not 'beam_size'"),
        ],
    )
    def test_refused(self, bench, capsys, argv, message):
        with pytest.raises(SystemExit):
            bench.run.parse_args(argv)

        assert message in capsys.readouterr().err

    def test_an_unknown_option_fails_before_anything_is_downloaded(self, bench, tmp_path):
        argv = ["-b", "faster-whisper", "-O", "beams=2", "--sets", "librispeech", "--limit", "1"]

        with pytest.raises(SystemExit, match="^Error: the faster-whisper backend has no option 'beams'"):
            bench.run.main([*argv, "--data-dir", str(tmp_path / "nothing")])

        assert not (tmp_path / "nothing").exists()

    def test_measured_with_the_options_and_kept_in_the_results(self, bench, monkeypatch, tmp_path):
        from benchmarks.data import Clip
        from speech_toolkit.backends import _BACKENDS

        monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {**_BACKENDS, "beams": BeamsBackend})
        monkeypatch.setattr(bench.run, "load_audio", lambda path: np.zeros(RATE, np.float32))
        clips = [Clip("ls/a", "ls", "en", "file", "a.wav", "0" * 64, 1.0, "beams 2")]
        messages: List[str] = []

        measured = bench.run.measure(
            "beams", "small", None, clips, tmp_path, log=messages.append, options={"beam_size": 2}
        )

        assert measured["options"] == {"beam_size": 2}
        assert measured["clips"]["ls/a"]["text"] == "beams 2"
        assert messages[0].startswith("beams small (beam_size=2): loaded in")

    def test_the_options_are_part_of_what_is_compared(self, bench):
        before = results(results_entry(None), results_entry({"beam_size": 1}, rtfx=30.0))
        after = results(results_entry({}), results_entry({"beam_size": 1}, rtfx=31.0))

        comparison = bench.compare.compare(before, after)

        assert not comparison.failed  # results without options compare with results with none
        assert [(line.model, line.options, line.rtfx) for line in comparison.lines] == [
            ("tiny", "", 20.0),
            ("tiny", "beam_size=1", 31.0),
        ]
        assert "| faster-whisper | tiny (beam_size=1) | cpu | librispeech |" in bench.compare.markdown(comparison)

        unmatched = bench.compare.compare(before, results(results_entry({}), results_entry({"beam_size": 2})))
        assert unmatched.problems == [
            "faster-whisper tiny (beam_size=1) on cpu is in the baseline but wasn't measured",
            "faster-whisper tiny (beam_size=2) on cpu has no baseline; add it (see benchmarks/README.md)",
        ]

    def test_options_label(self, bench):
        label = bench.compare.options_label

        assert label({"beam_size": 1, "batch_size": 8, "condition_on_previous_text": False}) == (
            "batch_size=8, beam_size=1, condition_on_previous_text=false"
        )
        assert label({}) == label(None) == ""

    def test_the_table_shows_them_with_the_model(self, bench):
        entry = {
            **results_entry({"batch_size": 8}),
            "gpu": None,
            "load_seconds": 1.0,
            "audio_seconds": 100.0,
            "peak_rss_bytes": None,
            "peak_gpu_bytes": None,
            "peak_gpu_sampled": False,
        }
        environment = {"machine": "Linux", "cpu": "Xeon", "cpus": 4, "python": "3.12", "speech_toolkit": "0.5"}

        table = bench.run.markdown({"environment": {**environment, "packages": {}}, "results": [entry]})

        assert "| faster-whisper | tiny (batch_size=8) | cpu | int8 | 1.0 | 100 | 20.0 |" in table
