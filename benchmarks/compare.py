"""Compare benchmark results with a baseline, and fail when a word error rate rose by more than the tolerance.

    python -m benchmarks.compare benchmarks/baseline.json benchmarks/results/20261027-120000.json
    python -m benchmarks.compare before.json after.json --max-wer-increase 0.005 --max-rtfx-drop 0.15

Prints a Markdown table to stdout: per model and set, the word error rate (WER) and speed (RTFx) of both runs. The exit
status is 1 when:

- a set's WER rose by more than ``--max-wer-increase`` (an absolute difference; default 0.01, one percentage point);
- with ``--max-rtfx-drop``, a model's RTFx fell by more than that fraction. Off by default: on shared CI runners speed
  varies too much to gate on, so it is only reported;
- the two can't be compared: another results format, a set measured on other clips or references (its fingerprint
  differs), or a model (backend, model and device) measured on one side only.

Models are matched by backend, model and the device they ran on. Other versions of the scoring tools, another
precision or another machine are noted. Only the JSON files are read: no model library or benchmark dependency is
needed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmarks import SCHEMA

DEFAULT_MAX_WER_INCREASE = 0.01
CHANGED_CLIPS = 5  # clips listed per regressed set
_EPSILON = 1e-9  # WERs are rounded to 6 digits: a change of exactly the tolerance passes

Key = Tuple[str, str, str]  # backend, model, device


@dataclasses.dataclass
class Line:
    """One model's results on one set, before and after."""

    backend: str
    model: str
    device: str
    set: str
    wer_before: Optional[float]
    wer: Optional[float]
    rtfx_before: Optional[float]
    rtfx: Optional[float]
    verdict: str = "ok"  # "ok", "regressed" or "improved"


@dataclasses.dataclass
class Comparison:
    max_wer_increase: float
    max_rtfx_drop: Optional[float]
    lines: List[Line] = dataclasses.field(default_factory=list)
    regressions: List[str] = dataclasses.field(default_factory=list)  # beyond the tolerance
    problems: List[str] = dataclasses.field(default_factory=list)  # results that can't be compared
    notes: List[str] = dataclasses.field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.regressions or self.problems)


def _key(entry: Dict[str, Any]) -> Key:
    return entry["backend"], entry["model"], entry["device"]


def _label(key: Key) -> str:
    return f"{key[0]} {key[1]} on {key[2]}"


def compare(
    baseline: Dict[str, Any],
    results: Dict[str, Any],
    max_wer_increase: float = DEFAULT_MAX_WER_INCREASE,
    max_rtfx_drop: Optional[float] = None,
) -> Comparison:
    """Compare *results* with *baseline*, both as python -m benchmarks.run writes them."""
    comparison = Comparison(max_wer_increase, max_rtfx_drop)
    for name, document in (("baseline", baseline), ("results", results)):
        if document.get("schema") != SCHEMA:
            comparison.problems.append(
                f"the {name} file is in results format {document.get('schema')!r}; this compares format {SCHEMA}"
            )
    if comparison.problems:
        return comparison
    scored = baseline["settings"].get("scoring"), results["settings"].get("scoring")
    if scored[0] != scored[1]:  # a new version may normalize a few words differently: a big change still fails
        comparison.notes.append(
            f"the WER was scored with {scored[0]} for the baseline but {scored[1]} now, which alone may change it"
        )
    before = {_key(entry): entry for entry in baseline["results"]}
    after = {_key(entry): entry for entry in results["results"]}
    if not after:
        comparison.problems.append("the results have no models")
    for key in before:
        if key not in after:
            comparison.problems.append(f"{_label(key)} is in the baseline but wasn't measured")
    for key in after:
        if key not in before:
            comparison.problems.append(f"{_label(key)} has no baseline; add it (see benchmarks/README.md)")
        else:
            _compare_model(comparison, key, before[key], after[key])
    _compare_machines(comparison, baseline, results)
    return comparison


def _compare_model(comparison: Comparison, key: Key, old: Dict[str, Any], new: Dict[str, Any]) -> None:
    for name in list(dict.fromkeys([*old["sets"], *new["sets"]])):
        if name not in old["sets"] or name not in new["sets"]:
            side = "the baseline" if name in old["sets"] else "these results"
            comparison.problems.append(f"{_label(key)}: only {side} measured the {name} set")
            continue
        _compare_set(comparison, key, name, old, new)
    if old.get("precision") != new.get("precision"):
        comparison.notes.append(f"{_label(key)} ran in {new.get('precision')}, the baseline in {old.get('precision')}")
    drop = comparison.max_rtfx_drop
    if drop is not None and old.get("rtfx") and new.get("rtfx") is not None and new["rtfx"] < old["rtfx"] * (1 - drop):
        comparison.regressions.append(
            f"{_label(key)}: RTFx fell from {old['rtfx']:.1f} to {new['rtfx']:.1f}, by more than {drop:.0%}"
        )


def _compare_set(comparison: Comparison, key: Key, name: str, old: Dict[str, Any], new: Dict[str, Any]) -> None:
    old_set, new_set = old["sets"][name], new["sets"][name]
    if old_set["fingerprint"] != new_set["fingerprint"]:
        comparison.problems.append(
            f"{_label(key)}: the {name} set was measured on other clips or references than the baseline's "
            f"(fingerprint {new_set['fingerprint']}, not {old_set['fingerprint']}); measure the baseline again"
        )
        return
    line = Line(*key, name, old_set["wer"], new_set["wer"], old_set.get("rtfx"), new_set.get("rtfx"))
    comparison.lines.append(line)
    change = new_set["wer"] - old_set["wer"]
    tolerance = comparison.max_wer_increase
    if change > tolerance + _EPSILON:
        line.verdict = "regressed"
        comparison.regressions.append(
            f"{_label(key)}: the WER on {name} rose from {old_set['wer']:.4f} to {new_set['wer']:.4f} "
            f"(+{change:.4f}, more than {tolerance:.4f})"
        )
        comparison.regressions += _changed_clips(old, new, name)
    elif change < -tolerance - _EPSILON:
        line.verdict = "improved"
        comparison.notes.append(
            f"{_label(key)}: the WER on {name} fell from {old_set['wer']:.4f} to {new_set['wer']:.4f}; if that's "
            "expected, update the baseline"
        )


def _changed_clips(old: Dict[str, Any], new: Dict[str, Any], name: str) -> List[str]:
    """The clips of set *name* whose errors rose most, as Markdown list items."""
    old_clips, new_clips = old.get("clips", {}), new.get("clips", {})
    rises = []
    for clip_id, found in new_clips.items():
        before = old_clips.get(clip_id)
        if found.get("set") == name and before is not None and found["errors"] > before["errors"]:
            rises.append((found["errors"] - before["errors"], clip_id, before, found))
    rises.sort(key=lambda rise: (-rise[0], rise[1]))
    return [
        f"  - `{clip_id}`: {before['errors']} → {found['errors']} errors of {found['words']} words; was "
        f"{before['hypothesis']!r}, now {found['hypothesis']!r}"
        for _, clip_id, before, found in rises[:CHANGED_CLIPS]
    ]


def _compare_machines(comparison: Comparison, baseline: Dict[str, Any], results: Dict[str, Any]) -> None:
    machines = [
        tuple(document.get("environment", {}).get(field) for field in ("machine", "cpu", "cpus"))
        for document in (baseline, results)
    ]
    if machines[0] != machines[1]:
        comparison.notes.append(
            "the two ran on different machines, so their speeds don't compare: "
            + " and ".join(f"{machine[1]} ({machine[2]} CPUs)" for machine in machines)
        )


def _number(value: Optional[float], digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _change(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None:
        return "-"
    return f"{after - before:+.4f}"


def _ratio(before: Optional[float], after: Optional[float]) -> str:
    if not before or after is None:
        return "-"
    return f"{after / before - 1:+.0%}"


def markdown(comparison: Comparison) -> str:
    """The comparison as Markdown: the table, the tolerance, then what failed and the notes."""
    lines = ["## Benchmark comparison", ""]
    if comparison.lines:
        lines += [
            "| Backend | Model | Device | Set | WER before | WER now | Change | RTFx before | RTFx now | Change | |",
            "|---|---|---|---|--:|--:|--:|--:|--:|--:|---|",
        ]
        for line in comparison.lines:
            cells = [line.backend, line.model, line.device, line.set, _number(line.wer_before, 4), _number(line.wer, 4)]
            cells += [_change(line.wer_before, line.wer), _number(line.rtfx_before, 1), _number(line.rtfx, 1)]
            cells += [_ratio(line.rtfx_before, line.rtfx), line.verdict]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    tolerance = comparison.max_wer_increase
    speed = (
        "RTFx is only reported"
        if comparison.max_rtfx_drop is None
        else f"RTFx may fall by at most {comparison.max_rtfx_drop:.0%}"
    )
    points = f"{tolerance * 100:g} percentage point{'s' * (tolerance != 0.01)}"
    lines.append(f"Tolerance: a set's WER may rise by at most {tolerance:.4f} ({points}); {speed}.")
    for title, items in (
        ("Regressions", comparison.regressions),
        ("Can't compare", comparison.problems),
        ("Notes", comparison.notes),
    ):
        if items:
            lines += ["", f"### {title}", ""]
            lines += [item if item.startswith("  - ") else f"- {item}" for item in items]
    verdict = "FAILED" if comparison.failed else "passed"
    lines += ["", f"**Result: {verdict}.**"]
    return "\n".join(lines) + "\n"


def _read(path: pathlib.Path) -> Dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"Error: can't read {path}: {e}")
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        sys.exit(f"Error: {path} isn't a results file of python -m benchmarks.run")
    return document


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.compare",
        description="Compare benchmark results with a baseline; exit with 1 when a word error rate rose by more than "
        "the tolerance, or the two can't be compared.",
    )
    parser.add_argument(
        "baseline", type=pathlib.Path, help="the results to compare with, e.g. benchmarks/baseline.json"
    )
    parser.add_argument("results", type=pathlib.Path, help="the new results")
    parser.add_argument(
        "--max-wer-increase",
        type=float,
        default=DEFAULT_MAX_WER_INCREASE,
        metavar="WER",
        help="how much a set's WER may rise, as an absolute difference (default: %(default)s, one percentage point)",
    )
    parser.add_argument(
        "--max-rtfx-drop",
        type=float,
        metavar="FRACTION",
        help="also fail when a model's RTFx falls by more than this fraction, e.g. 0.15 (default: only report speed)",
    )
    args = parser.parse_args(argv)
    if args.max_wer_increase < 0 or (args.max_rtfx_drop is not None and not 0 <= args.max_rtfx_drop < 1):
        parser.error("--max-wer-increase must be at least 0, and --max-rtfx-drop between 0 and 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    comparison = compare(_read(args.baseline), _read(args.results), args.max_wer_increase, args.max_rtfx_drop)
    sys.stdout.write(markdown(comparison))
    if comparison.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
