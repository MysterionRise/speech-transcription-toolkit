"""MkDocs hook: the docs pages render on the site as they do on GitHub, where they are written to be read too.

- **Lists after a line of text.** On GitHub a list may start right after a paragraph:

      How batch mode works:
      - **Folders** are searched recursively.

  Python-Markdown, which MkDocs uses, needs a blank line before the list, or it renders the items as part of the
  paragraph. This adds the blank line, outside code blocks. (``mdx_truly_sane_lists``, in mkdocs.yml, does the other
  half: lists nested with two spaces, as GitHub nests them.)
- **Heading ids** are made as GitHub makes them, so that links such as ``backends.md#adding-a-backend`` work on both.
  This is set here rather than in mkdocs.yml, where it would take a ``!!python`` tag that the check-yaml pre-commit
  hook refuses.

mkdocs.yml loads it under ``hooks``; only the docs build uses it.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from markdown import Extension, Markdown
from markdown.preprocessors import Preprocessor
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from pymdownx.slugs import slugify

LIST_ITEM = re.compile(r" *(?:[*+-]|(?P<number>\d{1,9})[.)])(?: +\S|$)")
FENCE = re.compile(r" *(?P<fence>`{3,}|~{3,})")
HEADING = re.compile(r" {0,3}#{1,6}(?:\s|$)")


class BlankLineBeforeLists(Preprocessor):
    """Inserts a blank line before a list that starts right after a paragraph, as CommonMark allows."""

    def run(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        fence: Optional[str] = None  # the open code fence, if inside one
        in_list = False  # whether the lines since the last blank line belong to a list
        previous = ""
        for line in lines:
            opening = FENCE.match(line)
            if fence is not None:
                fence = None if _closes(line, fence) else fence
            elif opening is not None:
                fence = opening["fence"]
            elif not line.strip() or HEADING.match(line):  # either ends a list
                in_list = False
            else:
                item = LIST_ITEM.match(line)
                if item and not in_list and previous.strip() and _interrupts_paragraph(item):
                    result.append("")
                in_list = in_list or item is not None
            result.append(line)
            previous = line
        return result


def _closes(line: str, fence: str) -> bool:
    """Whether *line* closes the code block that *fence* opened: the same character, at least as many, nothing else."""
    match = FENCE.match(line)
    if match is None:
        return False
    found = match["fence"]
    return found[0] == fence[0] and len(found) >= len(fence) and not line[match.end() :].strip()


def _interrupts_paragraph(item: "re.Match[str]") -> bool:
    """Whether the list item can start a list right after text: a bullet, or a numbered list from 1 (CommonMark)."""
    return item["number"] is None or int(item["number"]) == 1


class GitHubLists(Extension):
    def extendMarkdown(self, md: Markdown) -> None:  # noqa: N802 - Python-Markdown's name
        # Before fenced code is taken out (25), so that it sees the same lines as the other list processors.
        md.preprocessors.register(BlankLineBeforeLists(md), "blank_line_before_lists", 26)


@event_priority(50)  # before mkdocstrings reads the extensions, so that docstrings get them too
def on_config(config: MkDocsConfig, **kwargs: Any) -> MkDocsConfig:
    config.markdown_extensions.append(GitHubLists())  # type: ignore[arg-type]
    config.mdx_configs.setdefault("toc", {})["slugify"] = slugify(case="lower")  # GitHub's heading ids
    return config
