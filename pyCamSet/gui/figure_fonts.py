'''Purpose: Offer figures only open-source typefaces with a recorded licence.
Status: Active; the same approved families as the lab's optical-mapping GUI.
Future: Add a family only with its licence record and an upstream source.
'''
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ApprovedFont:
    """One approved family and the evidence behind its licence claim.

    ``bundled`` families ship inside Matplotlib with their licence text, so
    they exist on every platform.  The others are used only when the host
    already has them installed; their licence is the one the publishing
    project declares at ``source``.
    """

    family: str
    kind: str
    spdx: str
    source: str
    bundled: bool


#: Offered in this order, bundled (always available) families first.
APPROVED_FONTS: tuple[ApprovedFont, ...] = (
    ApprovedFont("DejaVu Sans", "sans", "Bitstream-Vera",
                 "matplotlib mpl-data/fonts/ttf/LICENSE_DEJAVU", True),
    ApprovedFont("DejaVu Serif", "serif", "Bitstream-Vera",
                 "matplotlib mpl-data/fonts/ttf/LICENSE_DEJAVU", True),
    ApprovedFont("DejaVu Sans Mono", "mono", "Bitstream-Vera",
                 "matplotlib mpl-data/fonts/ttf/LICENSE_DEJAVU", True),
    ApprovedFont("STIXGeneral", "serif", "OFL-1.1",
                 "matplotlib mpl-data/fonts/ttf/LICENSE_STIX", True),
    ApprovedFont("Liberation Sans", "sans", "OFL-1.1",
                 "https://github.com/liberationfonts/liberation-fonts/blob/main/LICENSE", False),
    ApprovedFont("Liberation Serif", "serif", "OFL-1.1",
                 "https://github.com/liberationfonts/liberation-fonts/blob/main/LICENSE", False),
    ApprovedFont("Liberation Mono", "mono", "OFL-1.1",
                 "https://github.com/liberationfonts/liberation-fonts/blob/main/LICENSE", False),
    ApprovedFont("Noto Sans", "sans", "OFL-1.1",
                 "https://github.com/notofonts/noto-fonts/blob/main/LICENSE", False),
    ApprovedFont("Noto Serif", "serif", "OFL-1.1",
                 "https://github.com/notofonts/noto-fonts/blob/main/LICENSE", False),
    ApprovedFont("Noto Sans Mono", "mono", "OFL-1.1",
                 "https://github.com/notofonts/noto-fonts/blob/main/LICENSE", False),
    ApprovedFont("Open Sans", "sans", "OFL-1.1",
                 "https://github.com/googlefonts/opensans/blob/main/OFL.txt", False),
    ApprovedFont("Roboto", "sans", "Apache-2.0",
                 "https://github.com/googlefonts/roboto/blob/main/LICENSE", False),
    ApprovedFont("Lato", "sans", "OFL-1.1",
                 "https://www.latofonts.com/lato-free-fonts/", False),
    ApprovedFont("Source Sans Pro", "sans", "OFL-1.1",
                 "https://github.com/adobe-fonts/source-sans/blob/release/LICENSE.md", False),
    ApprovedFont("PT Sans", "sans", "OFL-1.1",
                 "https://github.com/google/fonts/blob/main/ofl/ptsans/OFL.txt", False),
    ApprovedFont("PT Serif", "serif", "OFL-1.1",
                 "https://github.com/google/fonts/blob/main/ofl/ptserif/OFL.txt", False),
    ApprovedFont("Latin Modern Roman", "serif", "GUST-Font-License",
                 "https://www.gust.org.pl/projects/e-foundry/latin-modern", False),
)

_BY_NAME = {font.family.casefold(): font for font in APPROVED_FONTS}

#: The bundled family that stands in for each kind of typeface.
FALLBACKS = {"sans": "DejaVu Sans", "serif": "DejaVu Serif", "mono": "DejaVu Sans Mono"}

#: Common proprietary or unverified families, mapped to the kind of approved
#: family that replaces them, so an older saved serif style stays serif.
_KIND_HINTS = {
    "times new roman": "serif", "times": "serif", "georgia": "serif",
    "garamond": "serif", "palatino": "serif", "palatino linotype": "serif",
    "cambria": "serif", "book antiqua": "serif", "courier new": "mono",
    "courier": "mono", "consolas": "mono", "menlo": "mono",
}


def approved_font(family: str | None) -> ApprovedFont | None:
    """Return the approved record for *family*, or None if it is not approved."""
    if not family:
        return None
    return _BY_NAME.get(family.strip().casefold())


@lru_cache(maxsize=None)
def _installed(family: str) -> bool:
    """Whether Matplotlib has a real font file for *family* on this host."""
    from matplotlib import font_manager

    try:
        path = font_manager.findfont(font_manager.FontProperties(family=family),
                                     fallback_to_default=False)
    except (ValueError, RuntimeError):
        return False
    return bool(path)


def available_fonts() -> tuple[str, ...]:
    """The approved families that can actually render here, in offer order."""
    return tuple(font.family for font in APPROVED_FONTS
                 if font.bundled or _installed(font.family))


def resolve_figure_font(family: str | None) -> str | None:
    """Map a requested family onto an approved, installed one.

    ``None`` (the theme default) stays ``None``.  An approved family that is
    installed is kept; anything else -- a proprietary family from an older
    saved style, or an approved one missing on this host -- becomes the
    bundled family of the same kind, so figures and exports never depend on
    an unlicensed or absent font.
    """
    if family is None:
        return None
    record = approved_font(family)
    if record is not None and (record.bundled or _installed(record.family)):
        return record.family
    kind = record.kind if record is not None else _KIND_HINTS.get(family.strip().casefold(), "sans")
    return FALLBACKS[kind]
