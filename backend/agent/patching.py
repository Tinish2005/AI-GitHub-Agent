"""Apply one file's part of a unified diff to the file's real text.

Used when a pull request is created: the new file content must be exactly the
reviewed diff applied to the real file. The rules match the validation step:
every context and removed line must match the real file (the line numbers in
the diff may be a little off, like the `patch` tool allows).

Unlike the validation step, the original lines are kept as they are (including
Windows line endings), and added lines use the file's own line ending.
"""

from __future__ import annotations

from backend.agent.validation_pipeline import PatchApplyError, ValidationPipeline

_NO_NEWLINE_MARKER = "\\"


def dominant_eol(text: str) -> str:
    """"\\r\\n" if most lines end with it, otherwise "\\n"."""
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _with_eol(value: str, eol: str) -> str:
    return value.rstrip("\r\n") + eol


def apply_patch_to_text(text: str, patched_file, rel: str) -> str:
    """Return `text` with the hunks of `patched_file` applied, or raise PatchApplyError."""
    lines = text.splitlines(keepends=True)
    eol = dominant_eol(text)
    result: list = []
    cursor = 0

    def end_open_line() -> None:
        # A last line without a line ending must get one before more lines follow it.
        if result and not result[-1].endswith(("\n", "\r")):
            result[-1] = result[-1] + eol

    for hunk in patched_file:
        wanted = [ln.value for ln in hunk if ln.is_context or ln.is_removed]
        if hunk.source_length == 0:
            expected = hunk.source_start
        else:
            expected = max(hunk.source_start - 1, 0)
        pos = ValidationPipeline._locate(lines, wanted, expected, cursor)
        if pos is None:
            raise PatchApplyError(
                ValidationPipeline._mismatch(rel, hunk, lines, wanted, expected)
            )

        result.extend(lines[cursor:pos])
        used = 0
        last_was_added = False
        for line in hunk:
            if line.is_context:
                end_open_line()
                result.append(lines[pos + used])
                used += 1
                last_was_added = False
            elif line.is_removed:
                used += 1
                last_was_added = False
            elif line.is_added:
                end_open_line()
                result.append(_with_eol(line.value, eol))
                last_was_added = True
            elif getattr(line, "line_type", None) == _NO_NEWLINE_MARKER:
                # "\ No newline at end of file" after an added line.
                if last_was_added and result:
                    result[-1] = result[-1].rstrip("\r\n")
        cursor = pos + len(wanted)

    # Keep the rest of the file. If the last change left an open line, close it first.
    if cursor < len(lines):
        end_open_line()
    result.extend(lines[cursor:])
    return "".join(result)
