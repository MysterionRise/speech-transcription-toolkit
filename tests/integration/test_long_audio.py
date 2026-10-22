"""Audio longer than 30 s, for the backends and models CI runs: the samples joined into one 104 s clip.

Chunked backends cut it into several chunks. The timestamps must stay in order across the chunk boundaries, every
chunk's text must arrive, the word error rate must stay within the backend's threshold, and no backend may warn that
it cut text off. The full tier (test_full_tier.py) runs the same checks with the larger models.
"""

from __future__ import annotations

import pytest

from tests.integration.helpers import BACKEND_ARGS, check_long_audio, require


@pytest.mark.parametrize("backend", list(BACKEND_ARGS))
def test_long_audio(backend, long_clip, transcribe, tmp_path):
    require(backend)

    check_long_audio(backend, BACKEND_ARGS[backend], long_clip, transcribe, tmp_path)
