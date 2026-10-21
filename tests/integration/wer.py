"""Word error rate (WER), without extra dependencies.

WER is the number of word substitutions, deletions and insertions that turn the reference into the hypothesis,
divided by the number of words in the reference. Both texts are normalized first: lower case, no punctuation. 0.0 is
a perfect transcript; 1.0 means as many errors as the reference has words (an empty transcript, for example).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Sequence

# Runs of anything but letters, digits and apostrophes separate words ("speech-to-text" is three words).
_SEPARATORS = re.compile(r"[^\w']+|_+")


def normalize(text: str) -> List[str]:
    """The words of *text*, lower case and without punctuation: "Hello, world!" -> ["hello", "world"]."""
    text = unicodedata.normalize("NFKC", text).replace("’", "'").lower()  # ’ is an apostrophe too
    words = (word.strip("'") for word in _SEPARATORS.split(text))  # quotes go, "don't" stays one word
    return [word for word in words if word]


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """The fewest word substitutions, deletions and insertions that turn *reference* into *hypothesis*."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    """The word error rate of *hypothesis* against *reference*, after normalizing both."""
    ref_words = normalize(reference)
    if not ref_words:
        raise ValueError("the reference has no words")
    return edit_distance(ref_words, normalize(hypothesis)) / len(ref_words)
