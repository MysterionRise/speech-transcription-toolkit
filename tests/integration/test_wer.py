"""The WER helper the integration tests use for their accuracy thresholds."""

from __future__ import annotations

import pytest

from tests.integration.wer import edit_distance, normalize, wer


class TestNormalize:
    def test_case_and_punctuation(self):
        assert normalize("Hello world, this is a TEST.") == ["hello", "world", "this", "is", "a", "test"]

    def test_hyphens_and_other_symbols_separate_words(self):
        assert normalize("speech-to-text… (offline)!") == ["speech", "to", "text", "offline"]

    def test_apostrophes_stay_inside_words(self):
        assert normalize("Don’t say 'never'") == ["don't", "say", "never"]

    def test_no_words(self):
        assert normalize(" ... ") == []


class TestEditDistance:
    @pytest.mark.parametrize(
        "hypothesis, errors",
        [
            ("a b c d", 0),
            ("a x c d", 1),  # substitution
            ("a b d", 1),  # deletion
            ("a b c d e", 1),  # insertion
            ("b a c d", 2),
            ("", 4),
        ],
    )
    def test_counts_word_errors(self, hypothesis, errors):
        assert edit_distance("a b c d".split(), hypothesis.split()) == errors


class TestWer:
    def test_perfect_transcript(self):
        assert wer("Hello world. This is a test.", "hello world, this is a test") == 0

    def test_glued_words_count_twice(self):
        """A substitution and a deletion: what a Parakeet word-joining bug looks like (#27)."""
        assert wer("Hello world. This is a test.", "Helloworld. This is a test.") == pytest.approx(2 / 6)

    def test_empty_transcript(self):
        assert wer("Hello world.", "") == 1.0

    def test_insertions_can_exceed_one(self):
        assert wer("Hello.", "Hello there, how are you?") == 4.0

    def test_reference_needs_words(self):
        with pytest.raises(ValueError, match="no words"):
            wer("...", "Hello")
