"""Judge the education-mode release matrix (12 family x backend rows x Phases 1-4) from a real run.

target families and 12 family x backend rows, each phase reported as "image-corpus", "synthetic-render"
or "synthetic-detection" coverage, or untested; "a skip never counts as a pass".

Contract (the test files this tool runs):
  * ``tests/test_education_release_matrix_p1.py`` holds the Phase 1 cells and
    ``tests/test_education_release_matrix_p234.py`` the Phase 2-4 cells; each defines one test function
    ``test_release_cell`` parametrized with ids ``<Row>-p<N>`` (ids exactly as in ROWS; 12 + 36 = 48 cells);
  * a passing cell calls ``record_property("coverage", <kind>)`` with kind in COVERAGE;
  * a cell that cannot run calls ``pytest.skip("untested: <reason>")`` - never a silent pass.
This tool runs those files and checks the contract. The interpreter is ``--python`` (default: the one
running this script). Any interpreter other than the current one is labelled ``custom`` in the output; the pull request
must say which interpreter the matrix ran on.

Usage:
    python tools/edu_release/release_matrix.py [--phases 1|234|1234] [--python PATH]  # run + check; exit 0 = PASS
    python tools/edu_release/release_matrix.py --table [--python PATH]                 # run all + print only the table rows
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tests as rt  # noqa: E402  (same folder; shares the env pinning)

ROWS = [
    "ChArUco-aruco1", "ChArUco-aruco2", "ChArUco2-aruco2",
    "Ccube-aruco1", "Ccube-aruco2", "Ccube2-aruco2",
    "CIco-aruco1", "CIco-aruco2", "CIco2-aruco2",
    "PuzzleBoard-puzzle_board", "PuzzleBoardCube-puzzle_board", "PuzzleBoardIco-puzzle_board",
]
CORPUS_ROWS = ("ChArUco-aruco1", "Ccube-aruco1")   # the only rows with a real image corpus in tests/test_data
COVERAGE = ("image-corpus", "synthetic-render", "synthetic-detection")
TEST_FILES = {"1": "tests/test_education_release_matrix_p1.py", "234": "tests/test_education_release_matrix_p234.py"}
MIN_PASSED = {"1": 8, "234": 22}      # of 12 and 36 cells; the rest must be honest 'untested:' skips


def run_matrix(groups: list[str], python: str | None = None) -> tuple[dict, str]:
    # The interpreter defaults to the one running this script; any other is labelled 'custom'.
    py = python or sys.executable
    env = "current" if Path(py).resolve() == Path(sys.executable).resolve() else "custom"
    # stderr, so that --table's stdout is only the table the docs paste.
    print(f"interpreter: {py} ({env})",
          file=sys.stderr)
    cenv = rt.child_env(env)
    problem = rt.preflight(py, env, cenv)
    if problem:
        raise SystemExit(rt.fail(problem))
    files = [TEST_FILES[g] for g in groups]
    absent = [f for f in files if not (rt.ROOT / f).exists()]
    if absent:
        raise SystemExit(rt.fail(f"{', '.join(absent)} does not exist yet"))
    junit = rt.short_dir("junit", env) / f"m{os.getpid()}.xml"
    log = rt.short_dir("logs", env) / f"m{os.getpid()}.log"
    cmd = [py, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rs", f"--junitxml={junit}",
           f"--basetemp={rt.short_dir('pt', env) / f'm{os.getpid()}'}", "--timeout=1200", *files]
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        subprocess.run(cmd, cwd=rt.ROOT, env=cenv, stdout=fh, stderr=subprocess.STDOUT)
    if not junit.exists():
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-10:]
        raise SystemExit(rt.fail("the matrix run wrote no JUnit report:\n" + "\n".join(tail)))
    cells: dict = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        name = case.get("name", "")
        if not name.startswith("test_release_cell["):
            continue
        cid = name[len("test_release_cell["):-1]
        props = {p.get("name"): p.get("value") for p in case.iter("property")}
        outcome, msg = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = case.find(tag)
            if el is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                text = (el.get("message") or el.text or "").strip()
                msg = text.splitlines()[0] if text else ""
                break
        cells.setdefault(cid, []).append((outcome, msg, props.get("coverage")))
    junit.unlink(missing_ok=True)
    probe = ("import cv2, scipy, sys, importlib.util as u; "
             "print(f'Python {sys.version.split()[0]}, OpenCV {cv2.__version__}, SciPy {scipy.__version__}, '"
             "f'aruco2 importable: {bool(u.find_spec(\"aruco2\"))}, '"
             "f'puzzle_board importable: {bool(u.find_spec(\"puzzle_board\"))}')")
    env_line = subprocess.run([py, "-c", probe], cwd=rt.ROOT, env=cenv, capture_output=True, text=True).stdout.strip()
    return cells, env_line


def label(outcome: str, msg: str, cov: str | None) -> str:
    if outcome == "passed":
        return cov or "passed (no coverage kind)"
    if outcome == "skipped":
        reason = msg.split("untested:", 1)[1].strip() if "untested:" in msg else msg
        return f"untested ({reason[:60]})" if reason else "untested"
    return outcome.upper()


def table(cells: dict) -> list[str]:
    rows = ["| Family | Backend | Phase 1 | Phase 2 | Phase 3 | Phase 4 |", "|---|---|---|---|---|---|"]
    for row in ROWS:
        fam, backend = row.split("-", 1)
        vals = []
        for n in range(1, 5):
            got = cells.get(f"{row}-p{n}")
            vals.append(label(*got[0]) if got else "MISSING")
        rows.append(f"| {fam} | {backend} | " + " | ".join(vals) + " |")
    return rows


def check(cells: dict, groups: list[str]) -> list[str]:
    phases = [int(ch) for g in groups for ch in g]
    expected = {f"{r}-p{n}" for r in ROWS for n in phases}
    problems = []
    missing = sorted(expected - set(cells))
    extra = sorted(c for c in set(cells) - expected if not any(c.endswith(f"-p{n}") for n in range(1, 5)))
    if missing:
        problems.append(f"{len(missing)} cells missing (e.g. {', '.join(missing[:4])}); ids must be <Row>-p<N>")
    if extra:
        problems.append(f"unexpected cell ids {extra[:4]}")
    dup = [c for c, v in cells.items() if len(v) > 1]
    if dup:
        problems.append(f"duplicate cells {dup[:4]}")
    passed = {g: 0 for g in groups}
    for cid in sorted(expected & set(cells)):
        outcome, msg, cov = cells[cid][0]
        group = "1" if cid.endswith("-p1") else "234"
        if outcome in ("failed", "error"):
            problems.append(f"{cid} {outcome}: {msg[:160]}")
        elif outcome == "skipped" and "untested:" not in msg.lower()[:30]:
            problems.append(f"{cid} skipped without an 'untested: <reason>' message ({msg[:80]!r})")
        elif outcome == "passed":
            passed[group] += 1
            if cov not in COVERAGE:
                problems.append(f"{cid} passed without record_property('coverage', one of {COVERAGE})")
            elif group == "1" and cid.rsplit("-p", 1)[0] in CORPUS_ROWS and cov != "image-corpus":
                problems.append(f"{cid}: a corpus row's Phase 1 must use the image corpus, got {cov}")
            elif group == "234" and cov == "synthetic-render":
                problems.append(f"{cid}: Phases 2-4 cannot be covered by a single-face render")
    for row in CORPUS_ROWS:
        for n in phases:
            got = cells.get(f"{row}-p{n}")
            if not got or got[0][0] != "passed":
                problems.append(f"{row}-p{n} must pass: this row has a real image corpus")
    if len(groups) == 2:
        for row in ROWS:
            if not any(cells.get(f"{row}-p{n}", [("", "", "")])[0][0] == "passed" for n in phases):
                problems.append(f"row {row} has no passing phase at all")
    for g in groups:
        total = 12 * len(g)
        print(f"Phase(s) {'/'.join(g)}: {passed[g]} of {total} cells pass")
        if passed[g] < MIN_PASSED[g]:
            problems.append(f"Phase(s) {'/'.join(g)}: only {passed[g]} of {total} cells pass; "
                            f"at least {MIN_PASSED[g]} required")
    return problems


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--phases", default="1234", choices=["1", "234", "1234"])
    ap.add_argument("--python", default=None, metavar="PATH",
                    help="interpreter to run the matrix with (default: the one running this script)")
    a = ap.parse_args(argv)
    groups = ["1", "234"] if (a.table or a.phases == "1234") else [a.phases]
    cells, env_line = run_matrix(groups, a.python)
    if a.table:
        print("\n".join(table(cells)))
        return 0
    print(env_line)
    print("\n".join(table(cells)))
    problems = check(cells, groups)
    if problems:
        for p in problems[:12]:
            print("  " + p)
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
