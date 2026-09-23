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
    parser.add_argument("--theme", choices=("Light", "Dark", "Sepia"), default="Light",
                        help="Matplotlib chrome theme propagated from the GUI")
    parser.add_argument(
        "--save-dir", type=Path, default=None,
        help="write the figures here as PNGs as well")
    parser.add_argument(
        "--no-show", action="store_true",
        help="write the figures without opening any window")
    parser.add_argument(
        "--png", type=Path, default=None,
        help="render the offscreen three-panel assessment to this file "
             "instead of drawing the figures")
    parser.add_argument("--figure-width-mm", type=float, default=160.0)
    parser.add_argument("--figure-dpi", type=int, default=150)
    parser.add_argument("--figure-formats", nargs="+", choices=("png", "svg", "pdf"), default=("png",))
    parser.add_argument("--matplotlib-only", action="store_true",
                        help="skip PyVista scenes for a 2D-only export request")
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

    if args.png is not None:
        # Off screen is not the same as safe: on macOS pyvista still builds a
        # vtkCocoaRenderWindow, which is exactly what must not happen inside
        # the GUI process.
        from pyCamSet.gui.assess_calibration import _build_o_results
        from pyCamSet.utils.visualisation import render_calibration_pyvista_png

        o_results = _build_o_results(cams)
        if o_results is None:
            print("That camset carries no calibration results, so there is "
                  "nothing to render.", file=sys.stderr)
            return 1
        ok, detail = render_calibration_pyvista_png(
            o_results, cams.calibration_handler, str(args.png))
        if not ok:
            print(detail, file=sys.stderr)
            return 1
        print(detail)
        return 0

    try:
        from pyCamSet.utils.visualisation import visualise_calibration
        results = {"x": cams.calibration_params, "err": cams.calibration_result}
        visualise_calibration(
            results, cams.calibration_handler, show=not args.no_show,
            save_dir=args.save_dir, theme_name=args.theme,
            figure_width_mm=args.figure_width_mm, figure_dpi=args.figure_dpi,
            figure_formats=tuple(args.figure_formats), matplotlib_only=args.matplotlib_only)
    except Exception as exc:
        print(f"Could not draw the calibration: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
