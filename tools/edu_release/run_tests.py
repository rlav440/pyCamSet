"""Run a pytest selection in a pinned, Windows-safe environment and judge it from its JUnit report.

check calls this instead of bare ``pytest`` because Windows has traps that made the
report false failures:

* a stray ``python`` on PATH may be a version whose numba aborts with
  ``Illegal IR, del found`` -> run an explicit interpreter (the current one by default);
* numba's on-disk cache under a long worktree path exceeds MAX_PATH (LongPathsEnabled=0) and
  raises ``FileNotFoundError`` -> point ``NUMBA_CACHE_DIR`` at a short private folder
  (under ``EDU_SHORT_TMP`` or the system temp folder);
* an environment may have an editable install of a *different* pyCamSet checkout -> refuse to
  judge unless ``pyCamSet`` imports from this checkout.


Usage (from the checkout root):
    python tools/edu_release/run_tests.py [options] -- <pytest args>
Options:
    --env current|cv4|lean         interpreter (default current = the one running this script); cv4 and lean
                                   read their path from PYCAMSET_EDU_ENV_CV4 / PYCAMSET_EDU_ENV_LEAN
    --python PATH                  any other interpreter (labelled "custom")
    --require SUBSTR               a node whose id contains SUBSTR must exist and PASS (repeatable)
    --expect-skip SUBSTR           a node whose id contains SUBSTR must exist and be SKIPPED (repeatable)
    --min-passed N                 at least N tests must pass (default 1)
    --no-skips                     any skip fails the check
    --collect MIN                  only collect (``--collect-only``); require exit 0 and >= MIN tests
    --timeout S                    per-test timeout when pytest-timeout is installed (default 900)
    --allow-fail SUBSTR            a failure whose node id contains SUBSTR is reported but tolerated (repeatable;
                                   only for failures that also occur on the unmodified upstream code in the same env)
    --tests NAMES                  comma-separated test module names (``test_foo`` -> tests/test_foo.py), appended to
                                   the pytest args.  Use it where a bare module name is easier than a path.
Exit code 0 only when every rule holds. The last line is always ``RESULT: PASS`` or ``RESULT: FAIL``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Alternative interpreters come only from environment variables (or ``--python``); the default is the
# interpreter running this script. Each is a path to a python executable with pyCamSet's dependencies.
#   PYCAMSET_EDU_ENV_CV4   an environment with OpenCV 4.11 (and PySide6)
#   PYCAMSET_EDU_ENV_LEAN  a PySide6-free environment (the declared Python floor, OpenCV 5)
ENV_VARS = {"cv4": "PYCAMSET_EDU_ENV_CV4", "lean": "PYCAMSET_EDU_ENV_LEAN"}
EXPECT = {"cv4": "4.11", "lean": "5."}  # OpenCV version prefix a named env must have; "current" has no pin


def resolve_interpreter(env: str, python: str | None) -> tuple[str, str]:
    """(label, interpreter path) for the chosen ``--env`` / ``--python``; ValueError when unset."""
    if python:
        return "custom", python
    if env == "current":
        return "current", sys.executable
    var = ENV_VARS[env]
    path = os.environ.get(var)
    if not path:
        raise ValueError(f"--env {env} needs the interpreter path in the environment variable {var} "
                         "(or pass --python PATH)")
    return env, path


def fail(msg: str) -> int:
    print(msg)
    print("RESULT: FAIL")
    return 1


def short_dir(kind: str, env: str) -> Path:
    tag = hashlib.sha1(str(ROOT).lower().encode()).hexdigest()[:10]
    # EDU_SHORT_TMP keeps cache paths short on Windows (MAX_PATH); the default is the system temp folder.
    base = Path(os.environ.get("EDU_SHORT_TMP") or tempfile.gettempdir())
    d = base / kind / f"{tag}-{env}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def child_env(env: str) -> dict:
    e = dict(os.environ)
    e["NUMBA_CACHE_DIR"] = str(short_dir("nbc", env))
    e["PYTHONUTF8"] = "1"
    e["QT_QPA_PLATFORM"] = "offscreen"
    e["PYVISTA_OFF_SCREEN"] = "true"
    e["MPLBACKEND"] = "Agg"
    for k in ("PYTHONPATH", "PYTHONHOME", "NUMBA_DISABLE_JIT"):
        e.pop(k, None)
    return e


def preflight(py: str, env: str, cenv: dict) -> str | None:
    code = ("import sys, pathlib, cv2, numpy, scipy, numba, pyCamSet\n"
            "print(pathlib.Path(pyCamSet.__file__).resolve())\n"
            "print(f'python {sys.version.split()[0]} opencv {cv2.__version__} numpy {numpy.__version__} '\n"
            "      f'scipy {scipy.__version__} numba {numba.__version__}')\n"
            "import importlib.util\n"
            "print('pyside6 ' + ('yes' if importlib.util.find_spec('PySide6') else 'no'))\n")
    r = subprocess.run([py, "-c", code], cwd=ROOT, env=cenv, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return "preflight import failed:\n" + "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
    lines = r.stdout.strip().splitlines()
    imported = Path(lines[0])
    if ROOT.resolve() not in imported.parents:
        return f"pyCamSet imports from {imported}, not from this checkout {ROOT} - fix sys.path / cwd"
    print(f"env {env}: {lines[1]}")
    if env not in EXPECT:  # the current or a caller's own interpreter: no OpenCV or PySide6 requirement
        return None
    want = EXPECT[env]
    if want and f"opencv {want}" not in lines[1]:
        return f"env {env} must have OpenCV {want}*, got: {lines[1]}"
    if env == "lean" and lines[2] != "pyside6 no":
        return "env lean must not have PySide6 installed (it is the PySide6-free run)"
    if env != "lean" and lines[2] != "pyside6 yes":
        return f"env {env} must have PySide6 installed"
    return None


def parse_junit(path: Path) -> list[tuple[str, str, str]]:
    """[(node_id, outcome, first_message_line)] with outcome in passed/failed/error/skipped."""
    root = ET.parse(path).getroot()
    out = []
    for case in root.iter("testcase"):
        cls, name = case.get("classname", ""), case.get("name", "")
        node = (cls.replace(".", "/") + ".py::" + name) if cls else name
        outcome, msg = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = case.find(tag)
            if el is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                text = (el.get("message") or el.text or "").strip()
                msg = text.splitlines()[0][:220] if text else ""
                break
        out.append((node, outcome, msg))
    return out


def main(argv: list[str]) -> int:
    if "--" in argv:
        i = argv.index("--")
        own, pytest_args = argv[:i], argv[i + 1:]
    else:
        own, pytest_args = argv, []
    ap = argparse.ArgumentParser(prog="run_tests.py")
    ap.add_argument("--env", default="current", choices=["current", *sorted(ENV_VARS)],
                    help="interpreter: current (default, the one running this script), or cv4 / lean, whose "
                         "paths come from PYCAMSET_EDU_ENV_CV4 / PYCAMSET_EDU_ENV_LEAN")
    ap.add_argument("--python", default=None, metavar="PATH",
                    help="run with this interpreter instead (labelled 'custom')")
    ap.add_argument("--require", action="append", default=[])
    ap.add_argument("--expect-skip", action="append", default=[])
    ap.add_argument("--min-passed", type=int, default=1)
    ap.add_argument("--no-skips", action="store_true")
    ap.add_argument("--collect", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tests", default="")
    ap.add_argument("--allow-fail", action="append", default=[])
    a = ap.parse_args(own)
    for name in [n.strip() for n in a.tests.split(",") if n.strip()]:
        path = ROOT / "tests" / f"{name}.py"
        if not path.exists():
            return fail(f"--tests: {path} does not exist")
        pytest_args.append(str(path.relative_to(ROOT)))
    if not pytest_args:
        return fail("nothing to run: give --tests NAMES and/or pytest arguments after --")
    try:
        env, py = resolve_interpreter(a.env, a.python)
    except ValueError as exc:
        return fail(str(exc))
    if not Path(py).exists():
        return fail(f"interpreter for env {env} not found: {py} "
                    "(pass --python PATH to run with another interpreter)")
    cenv = child_env(env)
    problem = preflight(py, env, cenv)
    if problem:
        return fail(problem)

    base = [py, "-m", "pytest", "-p", "no:cacheprovider"]
    if a.collect is not None:
        r = subprocess.run(base + ["--collect-only", "-q"] + pytest_args, cwd=ROOT, env=cenv,
                           capture_output=True, text=True, timeout=1800)
        text = r.stdout + r.stderr
        m = re.search(r"(\d+) tests? collected", text)
        n = int(m.group(1)) if m else 0
        errs = [ln for ln in text.splitlines() if ln.startswith("ERROR ")][:8]
        print(f"collect ({env}): exit {r.returncode}, {n} tests collected")
        for ln in errs:
            print("  " + ln[:200])
        if r.returncode != 0 or n < a.collect:
            if not errs:
                print("\n".join(text.strip().splitlines()[-8:]))
            return fail(f"collection must exit 0 with >= {a.collect} tests")
        print("RESULT: PASS")
        return 0

    junit = short_dir("junit", env) / f"r{os.getpid()}.xml"
    basetemp = short_dir("pt", env) / f"b{os.getpid()}"
    cmd = base + ["-q", "-rfEs", f"--junitxml={junit}", f"--basetemp={basetemp}"]
    has_timeout = subprocess.run([py, "-c", "import pytest_timeout"], env=cenv, capture_output=True).returncode == 0
    if has_timeout:
        cmd.append(f"--timeout={a.timeout}")
    log = short_dir("logs", env) / f"r{os.getpid()}.log"
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run(cmd + pytest_args, cwd=ROOT, env=cenv, stdout=fh, stderr=subprocess.STDOUT).returncode
    if not junit.exists():
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-12:]
        return fail(f"pytest wrote no JUnit report (exit {rc}); last output:\n" + "\n".join(tail))
    cases = parse_junit(junit)
    junit.unlink(missing_ok=True)
    count = {k: sum(1 for c in cases if c[1] == k) for k in ("passed", "failed", "error", "skipped")}
    print(f"pytest exit {rc}: {count['passed']} passed, {count['failed']} failed, {count['error']} errors, "
          f"{count['skipped']} skipped (log {log})")
    problems = []
    bad = [c for c in cases if c[1] in ("failed", "error")]
    tolerated = [c for c in bad if any(sub in c[0] for sub in a.allow_fail)]
    bad = [c for c in bad if c not in tolerated]
    if tolerated:
        print(f"  {len(tolerated)} tolerated failures (pre-existing upstream): "
              + ", ".join(c[0].split("::")[-1][:50] for c in tolerated[:6]))
    for node, outcome, msg in bad[:8]:
        print(f"  {outcome.upper()} {node}: {msg}")
    if bad:
        problems.append(f"{len(bad)} tests failed or errored")
    if rc not in (0, 1) and not bad and not tolerated:
        problems.append(f"pytest exit code {rc} (interrupted, usage or collection error)")
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:]))
    if count["passed"] < a.min_passed:
        problems.append(f"only {count['passed']} passed; at least {a.min_passed} required")
    skipped = [c for c in cases if c[1] == "skipped"]
    if a.no_skips and skipped:
        for node, _, msg in skipped[:5]:
            print(f"  SKIPPED {node}: {msg}")
        problems.append(f"{len(skipped)} skipped; this check allows no skips")
    for sub in a.require:
        hits = [c for c in cases if sub in c[0]]
        if not hits:
            problems.append(f"required test '{sub}' was not collected")
        elif any(c[1] != "passed" for c in hits):
            problems.append(f"required test '{sub}' did not pass ({', '.join(sorted({c[1] for c in hits}))})")
    for sub in a.expect_skip:
        hits = [c for c in cases if sub in c[0]]
        if not hits:
            problems.append(f"test '{sub}' expected to be skipped was not collected")
        elif any(c[1] != "skipped" for c in hits):
            problems.append(f"test '{sub}' must be SKIPPED in env {env}, got {sorted({c[1] for c in hits})}")
    if problems:
        return fail("; ".join(problems))
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    sys.exit(main(sys.argv[1:]))
