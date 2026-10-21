"""Check the education-view release support statement in docs/how-to/gui.md.

documented support statement in docs/how-to/gui.md naming each untested family x backend x OS row';
Linux and macOS are reported "untested (local-only policy)".

Rules (each failure is printed in one line):
  * a section headed exactly ``## Education view: release support statement``;
  * inside it, the release matrix table between ``<!-- release-matrix:start -->`` and
    ``<!-- release-matrix:end -->``, identical (row by row, whitespace-normalised) to a fresh
    ``release_matrix.py --table`` run (pass --no-matrix to skip the re-run while drafting);
  * required facts, each matched case-insensitively: see REQUIRED;
  * at least one reference to a repository file (``tools/...``, ``tests/...`` or ``pyCamSet/...``).
Usage: python tools/edu_release/check_gui_doc.py [--no-matrix]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "how-to" / "gui.md"
HEADING = "## Education view: release support statement"
REQUIRED = {
    "Linux untested statement": r"linux[^\n]{0,80}untested \(local-only policy\)|untested \(local-only policy\)[^\n]{0,80}linux",
    "macOS untested statement": r"macos[^\n]{0,80}untested \(local-only policy\)|untested \(local-only policy\)[^\n]{0,80}macos",
    "skip is not a pass": r"skip(ped)?[^\n.]{0,40}(is|are) not (a )?pass",
    "OpenCV 5 canonical environment": r"opencv 5",
    "OpenCV 4.11 environment-specific results": r"opencv 4\.11",
    "short NUMBA_CACHE_DIR advice": r"numba_cache_dir",
    "how to turn the education mode on": r"education_preferences\.json",
    "the mode flag key": r"education_enabled",
    "recording is opt-in and off by default": r"(off|unticked|not ticked) by default",
    "developer launcher": r"python -m pyCamSet\.gui\.education_dialog",
    "OpenCV major and SciPy recorded": r"scipy",
}


def section(text: str) -> str | None:
    i = text.find(HEADING + "\n")
    if i < 0:
        return None
    rest = text[i + len(HEADING):]
    m = re.search(r"\n## ", rest)
    return rest[: m.start()] if m else rest


def norm(rows: list[str]) -> list[str]:
    return [re.sub(r"\s+", " ", r.strip()) for r in rows if r.strip().startswith("|")]


def main(argv: list[str]) -> int:
    problems = []
    if not DOC.exists():
        print(f"{DOC} is missing")
        print("RESULT: FAIL")
        return 1
    sec = section(DOC.read_text(encoding="utf-8"))
    if sec is None:
        print(f"no section headed exactly '{HEADING}' in docs/how-to/gui.md")
        print("RESULT: FAIL")
        return 1

    for name, rx in REQUIRED.items():
        if not re.search(rx, sec, re.I):
            problems.append(f"missing: {name} (pattern {rx!r})")
    if not re.search(r"`(tools|tests|pyCamSet)/[A-Za-z0-9_./-]+`", sec):
        problems.append("no reference to a repository file (tools/, tests/ or pyCamSet/) in the section")
    m = re.search(r"<!-- release-matrix:start -->(.*?)<!-- release-matrix:end -->", sec, re.S)
    if not m:
        problems.append("no <!-- release-matrix:start --> ... <!-- release-matrix:end --> block")
    elif "--no-matrix" not in argv:
        r = subprocess.run([sys.executable, str(Path(__file__).with_name("release_matrix.py")), "--table"],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        live = norm(r.stdout.splitlines())
        doc = norm(m.group(1).splitlines())
        if r.returncode != 0 or len(live) < 14:
            problems.append("could not regenerate the matrix table:\n" + r.stdout[-600:])
        elif live != doc:
            diff = [f"    doc : {d}\n    live: {l}" for d, l in zip(doc + [""] * 20, live) if d != l][:4]
            problems.append("the documented matrix differs from a fresh run "
                            "(paste `python tools/edu_release/release_matrix.py --table`):\n" + "\n".join(diff))
    if problems:
        for p in problems:
            print(p)
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    sys.exit(main(sys.argv[1:]))
