"""
Show a calibration target, as a program of its own.

``Ccube.plot()`` and ``PuzzleBoardCube.plot()`` open a pyvista window;
``ChArUco.plot()`` and ``PuzzleBoard.plot()`` open a matplotlib one.  None
of them can be opened from inside the running Qt application -- see
:mod:`pyCamSet.gui.viewer_process` for what happens when they are -- so
the Create Target tab runs this instead.

Run it as::

    python -m pyCamSet.utils.visualise_target '{"target_type": "Ccube", ...}'
"""
from __future__ import annotations

import argparse
import json
import sys

#: The parameters each target is built from, in the order its builder
#: takes them.  Anything else in the payload is ignored, so the GUI can
#: pass the whole of what it collected.
TARGET_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "Ccube": ("n_points", "length", "aruco_dict", "marker_backend"),
    "ChArUco": ("num_squares_x", "num_squares_y", "square_size",
                "marker_fraction", "aruco_dict", "marker_backend"),
    "PuzzleBoard": ("num_squares_x", "num_squares_y", "square_size",
                    "start_x", "start_y", "paper_width", "paper_height",
                    "min_width"),
    "PuzzleBoardCube": ("n_points", "length", "min_width"),
}

_INTEGERS = {"n_points", "num_squares_x", "num_squares_y", "start_x",
             "start_y", "min_width"}
_FLOATS = {"length", "square_size", "marker_fraction", "paper_width",
           "paper_height"}


def build_target(params: dict):
    """
    Construct a target from what the Create Target tab collected.

    :param params: the target settings, including ``target_type``
    :return: the target
    """
    from pyCamSet.calibration_targets.create_Ccube import build_ccube
    from pyCamSet.calibration_targets.create_charuco import build_charuco
    from pyCamSet.calibration_targets.create_puzzleboard import build_puzzleboard
    from pyCamSet.calibration_targets.create_puzzleboard_cube import (
        build_puzzleboard_cube)

    builders = {
        "Ccube": build_ccube,
        "ChArUco": build_charuco,
        "PuzzleBoard": build_puzzleboard,
        "PuzzleBoardCube": build_puzzleboard_cube,
    }
    target_type = str(params.get("target_type", ""))
    if target_type not in builders:
        raise ValueError(
            f"Unknown target type {target_type!r}; expected one of "
            f"{sorted(builders)}.")

    kwargs = {}
    for name in TARGET_ARGUMENTS[target_type]:
        if name not in params:
            raise ValueError(f"A {target_type} needs {name}.")
        value = params[name]
        if name in _INTEGERS:
            value = int(value)
        elif name in _FLOATS:
            value = float(value)
        else:
            value = str(value)
        kwargs[name] = value
    return builders[target_type](**kwargs)


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
        params = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"Could not read the target settings: {exc}", file=sys.stderr)
        return 2
    if not isinstance(params, dict):
        print("The target settings must be a JSON object.", file=sys.stderr)
        return 2

    try:
        target = build_target(params)
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
