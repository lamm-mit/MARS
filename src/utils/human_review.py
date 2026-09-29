"""Human-in-the-loop review gate for System 1 output.

Writes a Markdown checklist (``human_review.md``) into a run's artifacts
directory, then polls it until a domain expert flips ``Status: PENDING`` to
``Status: APPROVED`` or a timeout expires.  The expert can uncheck constraints
to remove them, check properties to promote them to hard constraints
(promotion is copy-not-move: System 2's candidate proposal and KG grounding
only see properties, so a promoted item must stay in the property list),
delete property lines to drop them, edit any line's text in place, and add
free-text constraints in a dedicated section.

Stdlib-only so it can be unit-tested without loading the pipeline.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

_STATUS_RE = re.compile(r"^\s*status\s*:\s*(\w+)", re.IGNORECASE)
_CHECKBOX_RE = re.compile(r"^\s*-\s*\[\s*([xX]?)\s*\]\s*(.+?)\s*$")
_NUM_PREFIX_RE = re.compile(r"^\d+\.\s*")
_BULLET_RE = re.compile(r"^\s*-\s+(?!\[)(.+?)\s*$")
_ANNOTATION_RE = re.compile(r"\s*<-.*$")

_SECTION_HEADERS = {
    "hard constraints": "constraints",
    "extracted properties": "properties",
    "add new constraints": "added",
}


@dataclass
class ReviewResult:
    """Parsed expert edits from a review file."""

    constraints: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    promoted: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    constraints_removed: int = 0

    @property
    def final_constraints(self) -> List[str]:
        """Kept constraints, then promoted properties, then added free text.

        Skips exact duplicates so double-editing cannot repeat a constraint.
        """
        merged: List[str] = []
        for item in self.constraints + self.promoted + self.added:
            if item and item not in merged:
                merged.append(item)
        return merged


def write_review_file(
    path: str,
    query_name: str,
    constraints: List[str],
    properties: List[str],
) -> None:
    """Write a fresh PENDING review checklist, replacing any stale file."""
    # A leftover APPROVED file from a previous run must never auto-satisfy
    # this run (same precedent as the MCP stale-marker deletion).
    if os.path.exists(path):
        os.unlink(path)

    lines = [
        f"# Human Review - {query_name}",
        "",
        "Status: PENDING",
        "",
        "Instructions:",
        "- Change the Status line above to APPROVED when you are done.",
        '- Uncheck a box under "Hard constraints" to REMOVE that constraint.',
        '- Check a box under "Extracted properties" to PROMOTE it to a hard',
        "  constraint (it also stays in the property list).",
        "- Delete a property line entirely to REMOVE it from the property list.",
        "- You may edit the text of any line; the edited text is used as-is.",
        '- Add new constraints as "- " bullet lines under "Add new constraints".',
        "",
        "## Hard constraints - uncheck to REMOVE",
    ]
    for i, constraint in enumerate(constraints, 1):
        lines.append(f"- [x] {i}. {constraint}")
    lines += [
        "",
        "## Extracted properties - check to PROMOTE to hard constraint",
    ]
    for prop in properties:
        lines.append(f"- [ ] {prop}")
    lines += [
        "",
        "## Add new constraints",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_review_status(path: str) -> Optional[str]:
    """Return the uppercased Status value, or None if missing/unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = _STATUS_RE.match(line)
                if m:
                    return m.group(1).upper()
    except OSError:
        return None
    return None


def parse_review_file(path: str) -> Optional[ReviewResult]:
    """Parse the checklist edits.

    Returns None if the file is missing or neither the constraints nor the
    properties section header is present (treated as unparseable).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return None

    result = ReviewResult()
    section: Optional[str] = None
    seen_sections = set()

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            header = stripped.lstrip("# ").lower()
            section = None
            for prefix, name in _SECTION_HEADERS.items():
                if header.startswith(prefix):
                    section = name
                    seen_sections.add(name)
                    break
            continue
        if section is None:
            continue

        m = _CHECKBOX_RE.match(line)
        if m:
            checked = m.group(1).lower() == "x"
            text = _ANNOTATION_RE.sub("", m.group(2)).strip()
            if not text:
                continue
            if section == "constraints":
                text = _NUM_PREFIX_RE.sub("", text)
                if checked:
                    result.constraints.append(text)
                else:
                    result.constraints_removed += 1
            elif section == "properties":
                result.properties.append(text)
                if checked:
                    result.promoted.append(text)
            elif section == "added":
                result.added.append(text)
            continue

        if section == "added":
            m = _BULLET_RE.match(line)
            if m:
                text = _ANNOTATION_RE.sub("", m.group(1)).strip()
                if text:
                    result.added.append(text)

    if not seen_sections & {"constraints", "properties"}:
        return None
    return result


def wait_for_review(
    path: str,
    poll_interval: float,
    timeout: float,
) -> Optional[ReviewResult]:
    """Poll until Status is APPROVED, then parse.

    Returns None on timeout, or if the approved file is unparseable (the
    caller falls back to the original System 1 output).  A missing file is
    treated as still pending, so an editor's atomic-save window or an
    accidental deletion never crashes the run.
    """
    start = time.monotonic()
    deadline = start + timeout
    next_reminder = start + 60.0
    while True:
        if parse_review_status(path) == "APPROVED":
            result = parse_review_file(path)
            if result is None:
                print(
                    "WARNING: review file approved but unparseable; "
                    "proceeding with original System 1 output",
                    flush=True,
                )
            return result
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            return None
        if now >= next_reminder:
            # flush=True: stdout is block-buffered when piped, and without a
            # flush these reminders would never reach the terminal mid-wait.
            print(
                f"Still waiting for review "
                f"({(now - start) / 60:.0f} min elapsed, "
                f"{remaining / 60:.0f} min until timeout): {path}",
                flush=True,
            )
            next_reminder = now + 60.0
        time.sleep(min(poll_interval, remaining))
