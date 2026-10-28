"""What a backend's model loads with: the model, the device, and the options the backend declares.

- **Models:** besides the names in its list, a backend can load custom models: a Hugging Face repo id (``org/name``),
  a local folder or a local file, whichever its ``MODEL_SOURCES`` name. :func:`model_source` tells them apart from
  the name alone, without going online.
- **Devices:** :func:`parse_device` reads ``auto``, ``cpu``, ``cuda``, ``cuda:N`` and ``mps``, for every backend and
  for speaker diarization. :func:`check_device` also refuses the kinds a backend doesn't list in ``DEVICES``.
- **Options:** a backend declares its settings in ``OPTIONS``, a map of names to :class:`BackendOption` specs.
  :func:`parse_options` checks ``Transcriber(options={...})`` and ``transcribe -O key=value`` against them, and
  converts command-line text to each option's type.

Everything here is checked before a model loads, so a typo fails at once instead of after a download.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, cast

from ..errors import UnsupportedOptionError

###############################################################################
# Models
###############################################################################

# What a backend can list in MODEL_SOURCES, with the words messages and --list-models use unless it gives its own.
MODEL_SOURCE_NAMES = {
    "hub": "a Hugging Face repo id (org/name)",
    "directory": "a local model folder",
    "file": "a local model file",
}

# A Hugging Face repo id: org/name, each starting and ending with a letter, digit or underscore, as the Hub requires.
_REPO_PART = r"\w(?:[\w.-]*\w)?"
_REPO_ID = re.compile(f"{_REPO_PART}/{_REPO_PART}")


def model_source(name: str) -> Optional[str]:
    """What the model *name* is, if it isn't a backend's own name for a model.

    ``"directory"`` or ``"file"`` if it exists on this machine, ``"hub"`` if it has the shape of a Hugging Face repo
    id (``org/name``), else None. Nothing is downloaded or looked up online: a repo id that doesn't exist is the
    loader's to report.
    """
    if os.path.isdir(name):
        return "directory"
    if os.path.isfile(name):
        return "file"
    if _REPO_ID.fullmatch(name) and "--" not in name and ".." not in name:
        return "hub"
    return None


###############################################################################
# Devices
###############################################################################

# The kinds of device a backend can list in DEVICES. "cuda" also covers "cuda:N", the N-th GPU (from 0).
DEVICE_KINDS = ("cpu", "cuda", "mps")
_DEVICE = re.compile(r"(auto|cpu|cuda|mps)(?::(\d+))?")


@dataclasses.dataclass(frozen=True)
class Device:
    """A device to run a model on: *kind* ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``, and *index*, the GPU's
    number in ``"cuda:N"`` (None when not given). ``str()`` gives it back as text, such as ``"cuda:1"``."""

    kind: str
    index: Optional[int] = None

    def __str__(self) -> str:
        return self.kind if self.index is None else f"{self.kind}:{self.index}"

    @property
    def auto(self) -> bool:
        """Whether the backend picks the device: ``"auto"``, or None."""
        return self.kind == "auto"


def parse_device(device: object) -> Device:
    """Parse a device: None or ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"`` (the N-th GPU, from 0) or ``"mps"``
    (an Apple GPU). Case and surrounding spaces don't matter, and a ``torch.device`` works too.

    Raises:
        UnsupportedOptionError: For anything else.
    """
    if device is None:
        return Device("auto")
    match = _DEVICE.fullmatch(str(device).strip().lower())
    if match is None or (match.group(2) is not None and match.group(1) != "cuda"):
        raise UnsupportedOptionError(f"unknown device {device!r}; use auto, cpu, cuda, cuda:N (N = 0, 1, ...) or mps")
    kind, index = match.groups()
    return Device(kind, None if index is None else int(index))


def describe_devices(kinds: Iterable[str]) -> str:
    """The devices a backend takes, for messages: ``"auto, cpu, cuda or cuda:N"`` for ``{"cpu", "cuda"}``."""
    names = ["auto"]
    for kind in DEVICE_KINDS:
        if kind in kinds:
            names += ["cuda", "cuda:N"] if kind == "cuda" else [kind]
    return f"{', '.join(names[:-1])} or {names[-1]}"


def check_device(device: object, kinds: Iterable[str], backend: str) -> Device:
    """:func:`parse_device` for a backend that runs on the device *kinds* (``"auto"`` always works).

    Raises:
        UnsupportedOptionError: If *device* isn't a device, or not one of *kinds*.
    """
    parsed = parse_device(device)
    kinds = frozenset(kinds)
    if not parsed.auto and parsed.kind not in kinds:
        raise UnsupportedOptionError(
            f"the {backend} backend can't run on {parsed.kind}; the devices it takes: {describe_devices(kinds)}"
        )
    return parsed


###############################################################################
# Options
###############################################################################

_TYPES = (bool, int, float, str)
_TRUE, _FALSE = ("true", "yes", "on", "1"), ("false", "no", "off", "0")


@dataclasses.dataclass(frozen=True)
class BackendOption:
    """A setting a backend declares in ``OPTIONS``. Users set it with ``Transcriber(options={name: value})`` or
    ``transcribe -O name=value``, and the backend reads it from ``self.options`` once its model has loaded.

    Attributes:
        type: ``bool``, ``int``, ``float`` or ``str``. Text from the command line is converted to it.
        default: The value when the option isn't given.
        help: What it does, in a sentence, for ``--list-backends`` and the docs.
        choices: The values it takes, if only some are allowed.
        minimum: The smallest value it takes, for a number.
    """

    type: type
    default: Any
    help: str
    choices: Tuple[Any, ...] = ()
    minimum: Optional[float] = None

    def __post_init__(self) -> None:
        if self.type not in _TYPES:
            raise TypeError(f"BackendOption.type must be bool, int, float or str, not {self.type!r}")

    @property
    def metavar(self) -> str:
        """What the value looks like, for help text: its choices (``auto|int8``), or its type (``INT``)."""
        if self.choices:
            return "|".join(_text(choice) for choice in self.choices)
        return {bool: "true|false", int: "INT", float: "NUMBER", str: "TEXT"}[self.type]

    def describe(self) -> str:
        """Its help, followed by its default: ``"Threads on the CPU ... (default: 0)"``."""
        return f"{self.help} (default: {_text(self.default)})"

    def parse(self, value: Any, name: str = "option") -> Any:
        """*value* as this option's type: text, as ``-O`` gives it, is converted; any other value must have the type
        already (an ``int`` does for a ``float``).

        Raises:
            UnsupportedOptionError: If *value* doesn't fit: the message says what does.
        """
        if isinstance(value, str) and self.type is not str:
            value = self._from_text(value.strip(), name)
        elif self.type is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, self.type) or (self.type is not bool and isinstance(value, bool)):
            raise UnsupportedOptionError(f"option {name} takes {self.metavar}, not {value!r}")
        if self.choices and value not in self.choices:
            raise UnsupportedOptionError(f"option {name} takes {self.metavar}, not {value!r}")
        if self.minimum is not None and cast(float, value) < self.minimum:  # a number: only numbers have a minimum
            raise UnsupportedOptionError(f"option {name} must be at least {self.minimum:g}, not {value!r}")
        return value

    def _from_text(self, text: str, name: str) -> Any:
        if self.type is bool:
            if text.lower() in _TRUE + _FALSE:
                return text.lower() in _TRUE
            raise UnsupportedOptionError(f"option {name} takes true or false, not {text!r}")
        try:
            return self.type(text)
        except ValueError:
            raise UnsupportedOptionError(f"option {name} takes {self.metavar}, not {text!r}") from None


def parse_options(declared: Mapping[str, BackendOption], given: Mapping[str, Any], backend: str) -> Dict[str, Any]:
    """The *given* options checked against the *declared* ones, and converted to their types.

    Names may use ``-`` for ``_`` (``cpu-threads``). Only the given options are returned, not the defaults.

    Raises:
        UnsupportedOptionError: For an option the backend doesn't declare (the message lists the ones it does), or
            a value that doesn't fit.
    """
    parsed: Dict[str, Any] = {}
    for key, value in given.items():
        name = str(key).strip().replace("-", "_")
        if name not in declared:
            valid = ", ".join(declared)
            raise UnsupportedOptionError(
                f"the {backend} backend has no option {name!r}; "
                + (f"its options: {valid}" if valid else "it takes no options")
            )
        parsed[name] = declared[name].parse(value, name)
    return parsed


def _text(value: Any) -> str:
    """*value* as the command line writes it: ``true`` rather than ``True``."""
    return str(value).lower() if isinstance(value, bool) else str(value)
