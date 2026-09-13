"""
Show a calibration target, as a program of its own.

``Ccube.plot()`` and ``PuzzleBoardCube.plot()`` open a pyvista window;
``ChArUco.plot()`` and ``PuzzleBoard.plot()`` open a matplotlib one.  None
of them can be opened from inside the running Qt application -- see
:mod:`pyCamSet.gui.viewer_process` for what happens when they are -- so
the Create Target tab runs this instead.

Run it as::

    python -m pyCamSet.utils.visualise_target '{"type": "Ccube", ...}'
"""
from __future__ import annotations

import argparse
import json
import sys

from pyCamSet.calibration_targets.target_registry import build_target

def main(argv: list[str] | None = None) -> int:
    """
    Build a target from a JSON payload and show it.

    :param argv: the command line, defaulting to ``sys.argv[1:]``
    :return: the process exit status
    """
    parser = argparse.ArgumentParser(
        prog="python -m pyCamSet.utils.visualise_target",
        description="Show a calibration target in its own window.",
    )
    parser.add_argument(
        "params", help="the target settings, as a JSON object")
    args = parser.parse_args(argv)

    try:
        spec = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"Could not read the target settings: {exc}", file=sys.stderr)
        return 2
    if not isinstance(spec, dict):
        print("The target settings must be a JSON object.", file=sys.stderr)
        return 2

    try:
        target = build_target(spec)
    except Exception as exc:
        print(f"Could not build the target: {exc}", file=sys.stderr)
        return 1

    try:
        target.plot()
    except Exception as exc:
        print(f"Could not draw the target: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
