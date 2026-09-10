"""
Show a saved calibration, as a program of its own.

:func:`~pyCamSet.utils.visualisation.visualise_calibration` opens native
matplotlib and pyvista windows.  Both are fine from a script or a notebook,
and neither can be opened from inside a running Qt application: pyvista's
Cocoa render window drives ``[NSRunLoop runUntilDate:]``, a nested event
loop that re-enters Qt's own event delivery and repaints its widgets from
inside a slot Qt has not finished dispatching.  On macOS that reliably
segmentation faults in ``QMacCGContext``.

So the GUI runs this instead, as a separate process.  The visualisation
gets a process where it is the only thing owning the event loop, and the
GUI keeps its own.

Run it as::

    python -m pyCamSet.utils.visualise_camset path/to/run.camset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """
    Load a camset and show its calibration.

    :param argv: the command line, defaulting to ``sys.argv[1:]``
    :return: the process exit status
    """
    parser = argparse.ArgumentParser(
        prog="python -m pyCamSet.utils.visualise_camset",
        description="Show the calibration saved in a .camset file.",
    )
    parser.add_argument("camset", type=Path, help="the .camset file to show")
    parser.add_argument(
        "--save-dir", type=Path, default=None,
        help="write the figures here as PNGs as well")
    parser.add_argument(
        "--no-show", action="store_true",
        help="write the figures without opening any window")
    args = parser.parse_args(argv)

    if not args.camset.is_file():
        print(f"No such camset: {args.camset}", file=sys.stderr)
        return 2

    from pyCamSet.utils.saving import load_CameraSet

    try:
        cams = load_CameraSet(args.camset)
    except Exception as exc:
        print(f"Could not load {args.camset}: {exc}", file=sys.stderr)
        return 1

    if getattr(cams, "calibration_handler", None) is None:
        print("That camset carries no calibration handler, so there is "
              "nothing to draw.", file=sys.stderr)
        return 1
    if getattr(cams, "calibration_params", None) is None:
        print("That camset carries no calibration results, so there is "
              "nothing to draw.", file=sys.stderr)
        return 1

    try:
        cams.visualise_calibration(
            show=not args.no_show, save_dir=args.save_dir)
    except Exception as exc:
        print(f"Could not draw the calibration: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
