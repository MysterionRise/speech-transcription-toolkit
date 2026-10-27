"""The benchmark harness: word error rate, speed and peak memory of each backend and model on a fixed evaluation set.

It runs from a checkout and isn't packaged. See benchmarks/README.md, and the Benchmarks section of docs/backends.md.

- ``data.py``: the evaluation set listed in ``manifest.json``, downloaded once into a cache folder and checked;
- ``metrics.py``: the word error rate (Whisper's text normalizers, then jiwer) and peak memory;
- ``run.py``: ``python -m benchmarks.run -b faster-whisper -m tiny``, which prints a Markdown table and writes JSON;
- ``compare.py``: ``python -m benchmarks.compare BASELINE RESULTS``, which fails when the word error rate rose;
- ``make_manifest.py``: picks the clips and writes ``manifest.json`` (maintainers only).
"""

# The version of the results' JSON format (README.md describes it). Bump it when a field changes meaning or goes;
# python -m benchmarks.compare refuses to compare files of different versions.
SCHEMA = 1
