"""Griffe extension for the API reference: teaches griffe's Google-style parser two habits of the docstrings.

- **Sphinx roles,** such as :class:`Transcriber` or :meth:`~TranscriptionResult.load`, become links to the object when
  the reference documents it, and code when it doesn't. Names resolve as in Python, from the docstring's scope.
- **Parameters that share a description,** ``num_speakers, min_speakers, max_speakers: ...`` in ``Args``, get an
  entry each. Griffe would otherwise warn that no parameter has that name, which fails ``mkdocs build --strict``.

mkdocs.yml loads it in the mkdocstrings options. Only the docs build uses it, so speech_toolkit keeps its docstrings
as they are and imports nothing from here.
"""

from __future__ import annotations

import html
import re
from typing import Any, Iterator, List

import griffe

ROLE = re.compile(r":(?P<role>class|func|meth|attr|mod|data|exc|obj):`(?P<short>~?)(?P<path>[\w.]+)`")
CALLABLE_ROLES = ("func", "meth")  # Sphinx shows these with parentheses: to_dict()
LITERAL_BLOCK = re.compile(r"(?<=\S)::$", re.MULTILINE)  # "Like this::" introduces a code block in Sphinx
PARAMETER_SECTIONS = ("Args:", "Arguments:", "Parameters:", "Params:", "Keyword Args:", "Keyword Arguments:")
SHARED_ENTRY = re.compile(r"(?P<indent>[ \t]+)(?P<names>\w+(?:,[ \t]*\w+)+):(?P<rest>.*)")


class SphinxMarkup(griffe.Extension):
    """Rewrites every docstring of a package once it has loaded, before mkdocstrings parses them."""

    def on_package(self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any) -> None:
        for obj in _objects(pkg):
            if obj.docstring is not None:
                text = split_shared_parameters(obj.docstring.value)
                text = LITERAL_BLOCK.sub(":", text)
                obj.docstring.value = ROLE.sub(lambda match: _reference(match, obj, loader), text)


def _objects(obj: griffe.Object) -> Iterator[griffe.Object]:
    """*obj* and the objects defined in it, recursively; not imported names, whose docstrings are their own module's."""
    yield obj
    for member in obj.members.values():
        if not member.is_alias:
            yield from _objects(member)  # type: ignore[arg-type]


def _reference(match: "re.Match[str]", obj: griffe.Object, loader: griffe.GriffeLoader) -> str:
    """The HTML for one role: an optional cross-reference, which autorefs turns into a link if the target is shown."""
    path = match["path"]
    title = path.rpartition(".")[2] if match["short"] else path
    if match["role"] in CALLABLE_ROLES:
        title += "()"
    target = html.escape(_full_path(path, obj, loader), quote=True)
    return f'<autoref identifier="{target}" optional><code>{html.escape(title)}</code></autoref>'


def _full_path(path: str, obj: griffe.Object, loader: griffe.GriffeLoader) -> str:
    """The canonical path of the object *path* names in the docstring of *obj*, or *path* if it can't be found."""
    first, dot, rest = path.partition(".")
    try:
        path = obj.resolve(first) + dot + rest
    except griffe.NameResolutionError:
        pass  # a full path, such as speech_toolkit.Transcriber
    try:
        return loader.modules_collection[path].canonical_path
    except (KeyError, ValueError, griffe.AliasResolutionError, griffe.CyclicAliasError):
        return path


def split_shared_parameters(text: str) -> str:
    """*text* with each ``a, b: description`` entry of its parameter sections repeated for ``a`` and for ``b``."""
    lines = text.split("\n")
    result: List[str] = []
    in_parameters = False
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.strip() and not line[0].isspace():  # a section title, or prose
            in_parameters = line.strip() in PARAMETER_SECTIONS
        entry = SHARED_ENTRY.fullmatch(line) if in_parameters else None
        if entry is None:
            result.append(line)
            continue
        continuation = []  # the lines of the description after the first, indented further than the entry
        while index < len(lines) and _indent(lines[index]) > len(entry["indent"]) and lines[index].strip():
            continuation.append(lines[index])
            index += 1
        for name in entry["names"].split(","):
            result += [f"{entry['indent']}{name.strip()}:{entry['rest']}", *continuation]
    return "\n".join(result)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())
