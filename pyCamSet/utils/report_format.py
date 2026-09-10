"""
Shared layout for the blocks pyCamSet prints during a calibration.

A calibration run reports at three points -- what the input data looks like,
what the optimiser is doing, and what came out -- and those three only read as
one system if they are laid out by one set of rules. Everything here exists to
keep them to a single width, so nothing wraps in an 80 column terminal.
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
from dataclasses import dataclass

from humanfriendly.terminal import ansi_wrap

# The Solarized palette, as its canonical xterm-256 indices. Solarized picks
# its eight accents to hold contrast against both of its backgrounds, which is
# what both the log and these blocks need: nothing here may depend on the
# terminal being light or dark. 256 colour indices rather than 24 bit values
# because every terminal, tmux and screen included, can show them.
SOLARIZED = {
    "base01": 240,   # secondary content, readable on light and dark
    "blue": 33,
    "green": 64,
    "yellow": 136,
    "orange": 166,
    "red": 160,
}

# Three scales, all of them reading green as good and red as bad, so a glance
# down any table means the same thing.

#: Detection rates. At or above this fraction is good, and shown green.
GOOD_FRACTION = 0.90

#: Detection rates. At or above this fraction is workable, and shown orange.
#: Below it the value is shown red.
FAIR_FRACTION = 0.60

#: Rig deviations, in mm or degrees. Under this is a rigid rig, shown green.
DEVIATION_GOOD = 1.0

#: Rig deviations. Under this is tolerable, shown orange; at or above it the
#: value is shown red.
DEVIATION_FAIR = 5.0

#: Reprojection error. Under this is an exceptional calibration, shown blue.
ERROR_EXCELLENT_PX = 0.1

#: Reprojection error. Under this is a good calibration, shown green.
ERROR_GOOD_PX = 1.0

#: Reprojection error. Under this is workable, shown orange; at or above it
#: the value is shown red. Matches calibration_report.HIGH_FINAL_ERROR_PX.
ERROR_FAIR_PX = 5.0

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: The width of the rule under a title, and of the tables under it.
RULE_WIDTH = 66

#: Nothing may reach this column. An 80 column terminal wraps at 80, and a
#: wrapped line is exactly the problem these blocks exist to solve.
MAX_WIDTH = 78


@dataclass
class Cell:
    """
    A table cell that carries its own colour.

    The colour is applied after the cell has been padded to its column, so an
    escape sequence -- which occupies no columns but plenty of characters --
    can never shift the table it sits in.

    :param text: the visible text
    :param colour: an xterm-256 index, or None for the default foreground
    """
    text: str
    colour: int | None = None


def colour_enabled(override: bool | None = None) -> bool:
    """
    Whether to emit colour: only for a person at a terminal.

    Honours NO_COLOR (https://no-color.org) so that a run whose output is
    being captured can ask for plain text without redirecting it.

    :param override: force colour on or off, bypassing the detection
    """
    if override is not None:
        return override
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(sys.stderr is not None and sys.stderr.isatty())
    except (AttributeError, ValueError):     # closed or replaced stderr
        return False


def quality_colour(fraction: float) -> int:
    """
    The band a fraction falls in, as a colour.

    :param fraction: the value, where 1.0 is 100%
    """
    if fraction >= GOOD_FRACTION:
        return SOLARIZED["green"]
    if fraction >= FAIR_FRACTION:
        return SOLARIZED["orange"]
    return SOLARIZED["red"]


def quality_cell(fraction: float) -> Cell:
    """
    A percentage cell, coloured by the band it falls in.

    :param fraction: the value, where 1.0 is 100%
    """
    return Cell(percent(fraction), quality_colour(fraction))


def deviation_colour(value: float) -> int:
    """
    The band a rig deviation falls in, as a colour.

    The same numeric bands serve millimetres and degrees: under 1 is a rigid
    rig, under 5 is tolerable, and anything more says the images are not of
    the same instant.

    :param value: the deviation, in mm or in degrees
    """
    if value < DEVIATION_GOOD:
        return SOLARIZED["green"]
    if value < DEVIATION_FAIR:
        return SOLARIZED["orange"]
    return SOLARIZED["red"]


def deviation_cell(value: float, places: int = 2) -> Cell:
    """
    A rig deviation cell, coloured by the band it falls in.

    :param value: the deviation, in mm or in degrees
    :param places: decimal places to show
    """
    return Cell(f"{value:.{places}f}", deviation_colour(value))


def error_colour(pixels: float) -> int:
    """
    The band a reprojection error falls in, as a colour.

    Four bands rather than three, because the difference between a calibration
    that is fine and one that is genuinely excellent is worth seeing: under a
    tenth of a pixel is blue.

    A NaN error falls through every comparison to red, which is the right
    answer for a solve that did not produce a usable number.

    :param pixels: the error, in pixels
    """
    if pixels < ERROR_EXCELLENT_PX:
        return SOLARIZED["blue"]
    if pixels < ERROR_GOOD_PX:
        return SOLARIZED["green"]
    if pixels < ERROR_FAIR_PX:
        return SOLARIZED["orange"]
    return SOLARIZED["red"]


def error_cell(pixels: float, places: int = 2) -> Cell:
    """
    A reprojection error cell, coloured by the band it falls in.

    :param pixels: the error, in pixels
    :param places: decimal places to show
    """
    return Cell(f"{pixels:.{places}f}", error_colour(pixels))


def strip_ansi(text: str) -> str:
    """
    The text without its escape sequences, for measuring or comparing it.

    :param text: the text to strip
    """
    return _ANSI.sub("", text)


def title(text: str) -> list[str]:
    """
    A block title and its rule.

    :param text: the title
    """
    return [text, "-" * RULE_WIDTH]


def kv_rows(pairs: list[tuple[str, str]]) -> list[str]:
    """
    Key and value pairs, two to a line, as a loose grid.

    :param pairs: the (label, value) pairs, in reading order
    """
    lines = []
    for index in range(0, len(pairs), 2):
        k0, v0 = pairs[index]
        left = f"  {k0:<15}{v0:>10}"
        if index + 1 < len(pairs):
            k1, v1 = pairs[index + 1]
            lines.append(f"{left}     {k1:<17}{v1:>10}")
        else:
            lines.append(left)
    return lines


def table(headers: list[str], rows: list[list], widths: list[int],
          colour: bool | None = None) -> list[str]:
    """
    A table with a left aligned first column and right aligned numbers.

    Cells are truncated to their column, so no value -- a long camera name in
    particular -- can push the table past its width. A cell given as a
    :class:`Cell` is coloured, and is padded before it is coloured so that the
    escape sequences cost no columns.

    :param headers: the column headings
    :param rows: the rows, each as many cells as there are headers
    :param widths: the column widths, which must sum to at most RULE_WIDTH
    :param colour: force colour on or off, defaulting to auto detection
    """
    use_colour = colour_enabled(colour)

    def render_cell(cell, width: int, first: bool) -> str:
        text, shade = ((cell.text, cell.colour) if isinstance(cell, Cell)
                       else (str(cell), None))
        text = text[:width]
        padded = f"{text:<{width}}" if first else f"{text:>{width}}"
        return ansi_wrap(padded, color=shade) if (shade and use_colour) else padded

    def render(cells) -> str:
        return "  " + "".join(
            render_cell(cell, width, index == 0)
            for index, (cell, width) in enumerate(zip(cells, widths)))

    return ([render(headers), "  " + "-" * sum(widths)]
            + [render(row) for row in rows])


def flag_lines(flags: list[str]) -> list[str]:
    """
    The concerns at the foot of a block, each as its own wrapped paragraph.

    :param flags: the flag texts
    """
    lines = []
    for flag in flags:
        lines += [""] + wrap(flag)
    return lines


def wrap(text: str, lead: str = "  ! ", width: int = MAX_WIDTH) -> list[str]:
    """
    Wrap text into a block whose continuations line up under the first line.

    :param text: the text to wrap
    :param lead: the prefix for the first line, its width sets the indent
    :param width: the column to wrap at
    """
    body = textwrap.wrap(text, width=width - len(lead))
    if not body:
        return []
    pad = " " * len(lead)
    return [lead + body[0]] + [pad + line for line in body[1:]]


def wrap_items(lead: str, items: list[str], width: int = MAX_WIDTH) -> list[str]:
    """
    Lay a list of short items out over as many lines as it takes, breaking
    only between items so that no entry is ever split in half.

    :param lead: the prefix for the first line, its width sets the indent
    :param items: the already formatted items
    :param width: the column to wrap at
    """
    if not items:
        return []
    pad = " " * len(lead)
    lines, current = [], lead
    for index, item in enumerate(items):
        piece = item + ("," if index < len(items) - 1 else "")
        candidate = current + ("" if current in (lead, pad) else " ") + piece
        if len(candidate) < width or current in (lead, pad):
            current = candidate
        else:
            lines.append(current)
            current = pad + piece
    lines.append(current)
    return lines


def percent(fraction: float) -> str:
    """
    A fraction as a percentage, for a table cell.

    :param fraction: the value, where 1.0 is 100%
    """
    return f"{fraction * 100:.1f}%"
