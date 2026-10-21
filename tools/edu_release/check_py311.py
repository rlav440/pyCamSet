"""Prove every tracked Python file still parses on the project's declared floor (Python 3.11).

and the docs CI job runs 3.11, but the usual test interpreter is 3.12, which accepts PEP 701
f-strings (nested same-type quotes) that 3.11 rejects. This compiles each tracked ``.py`` under
``pyCamSet/``, ``tests/`` and ``tools/`` with a real 3.11 interpreter (no imports, no execution).

Usage: python tools/edu_release/check_py311.py [--python PATH]
  --python PATH   a Python 3.11 interpreter to use. Without it, the first of: this script's own
                  interpreter (if it is 3.11), the ``py -3.11`` launcher, ``python3.11``, ``python3``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHILD = r"""
import json, sys
bad = []
for path in json.loads(sys.stdin.read()):
    try:
        with open(path, encoding="utf-8") as fh:
            compile(fh.read(), path, "exec", dont_inherit=True)
    except SyntaxError as exc:
        bad.append(f"{path}:{exc.lineno}: {exc.msg}")
print(json.dumps({"version": sys.version.split()[0], "bad": bad}))
"""


def _is_311(cmd: list[str]) -> bool:
    """True when ``cmd`` (an interpreter command, possibly a launcher with arguments) is Python 3.11."""
    probe = "import sys; print(sys.version_info[:2] == (3, 11))"
    try:
        r = subprocess.run(cmd + ["-c", probe], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and r.stdout.strip() == "True"


def find_python311(explicit: str | None) -> list[str] | None:
    """The command for a Python 3.11 interpreter, or None when none is available."""
    if explicit:
        return [explicit] if _is_311([explicit]) else None
    candidates = [[sys.executable]]
    if shutil.which("py"):  # the Windows launcher; it may hold several versions
        candidates.append(["py", "-3.11"])
    for name in ("python3.11", "python3"):
        if shutil.which(name):
            candidates.append([name])
    return next((cmd for cmd in candidates if _is_311(cmd)), None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="check_py311.py")
    ap.add_argument("--python", default=None, metavar="PATH")
    a = ap.parse_args(argv)
    py = find_python311(a.python)
    if not py:
        print("no Python 3.11 interpreter found" + (f" at {a.python}" if a.python else "")
              + "; pass --python PATH to a Python 3.11 interpreter")
        print("RESULT: FAIL")
        return 1
    files = subprocess.run(["git", "ls-files", "--", "pyCamSet/*.py", "tests/*.py", "tools/*.py"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    files += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "tests").glob("*.py")
              if str(p.relative_to(ROOT)).replace("\\", "/") not in files]  # untracked new tests too
    r = subprocess.run(py + ["-c", CHILD], cwd=ROOT, input=json.dumps(files), capture_output=True, text=True,
                       timeout=600)
    if r.returncode != 0:
        print(r.stderr.strip()[-800:])
        print("RESULT: FAIL")
        return 1
    res = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"compiled {len(files)} files with Python {res['version']}")
    for line in res["bad"][:15]:
        print("  SYNTAX " + line)
    if res["bad"]:
        print(f"{len(res['bad'])} file(s) do not parse on Python 3.11 (declared floor in pyproject.toml); "
              "rewrite the offending f-strings/syntax without changing behaviour")
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
    sys.exit(main())
