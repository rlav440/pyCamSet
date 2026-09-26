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
import inspect
import json
import sys
from pathlib import Path

from pyCamSet.calibration_targets.core.target_registry import build_target

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
    parser.add_argument("--save-png", type=Path, help="save the rendered target figure/scene as PNG")
    parser.add_argument(
        "--save-geometry", type=Path,
        help="export genuine target scene geometry (format inferred from .ply/.obj/.gltf)",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow replacing existing export files")
    parser.add_argument("--3d-background", dest="three_d_background",
                        choices=("theme", "white", "charcoal"))
    parser.add_argument("--3d-point-size", dest="three_d_point_size", type=float)
    parser.add_argument("--3d-view", dest="three_d_view",
                        choices=("isometric", "top", "front", "side"))
    parser.add_argument("--3d-theme", dest="three_d_theme",
                        choices=("Light", "Dark", "Sepia"), default="Light")
    axes = parser.add_mutually_exclusive_group()
    axes.add_argument("--3d-axes", dest="three_d_axes", action="store_true")
    axes.add_argument("--no-3d-axes", dest="three_d_axes", action="store_false")
    legend = parser.add_mutually_exclusive_group()
    legend.add_argument("--3d-legend", dest="three_d_legend", action="store_true")
    legend.add_argument("--no-3d-legend", dest="three_d_legend", action="store_false")
    parser.set_defaults(three_d_axes=None, three_d_legend=None)
    args = parser.parse_args(argv)

    try:
        spec = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"Could not read the target settings: {exc}", file=sys.stderr)
        return 2
    if not isinstance(spec, dict):
        print("The target settings must be a JSON object.", file=sys.stderr)
        return 2

    style = {"background": args.three_d_background,
             "point_size": args.three_d_point_size, "view": args.three_d_view,
             "axes": args.three_d_axes, "legend": args.three_d_legend}
    style = {key: value for key, value in style.items() if value is not None}
    if style:
        style["theme"] = args.three_d_theme
    try:
        target = build_target(spec)
    except Exception as exc:
        print(f"Could not build the target: {exc}", file=sys.stderr)
        return 1

    if args.save_png or args.save_geometry:
        try:
            _export_target(target, args.save_png, args.save_geometry,
                           overwrite=args.overwrite, style=style)
        except Exception as exc:
            print(f"Could not export the target view: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        if "return_scene" in inspect.signature(target.plot).parameters:
            scene = target.plot(return_scene=True)
            _apply_target_style(scene, style)
            scene.show()
        elif style:
            raise ValueError("This target renderer does not support managed 3D styles.")
        else:
            target.plot()
    except Exception as exc:
        print(f"Could not draw the target: {exc}", file=sys.stderr)
        return 1
    return 0


def _export_target(
    target: object,
    png_path: Path | None,
    geometry_path: Path | None,
    *,
    overwrite: bool = False,
    style: dict | None = None,
) -> None:
    """Export only actual rendered figures or scene objects; refuse printable layouts."""
    output_paths = [path for path in (png_path, geometry_path) if path is not None]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("PNG and geometry exports must use different output paths.")
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing export: {existing[0]}; pass --overwrite to replace it.")

    scene = None
    style = style or {}
    plot_method = target.plot
    if "return_scene" in inspect.signature(plot_method).parameters:
        scene = plot_method(return_scene=True)
    if scene is not None and hasattr(scene, "screenshot"):
        _apply_target_style(scene, style)
        if geometry_path is not None:
            extension = geometry_path.suffix.lower()
            geometry_path.parent.mkdir(parents=True, exist_ok=True)
            if extension == ".gltf" and hasattr(scene, "export_gltf"):
                scene.export_gltf(str(geometry_path))
            elif extension in {".ply", ".obj"} and hasattr(scene, "meshes"):
                import pyvista as pv

                meshes = [mesh for mesh in scene.meshes if isinstance(mesh, pv.PolyData) and mesh.n_points]
                if not meshes:
                    raise ValueError("The renderer has no mesh geometry to export.")
                merged = meshes[0].merge(meshes[1:], merge_points=False) if len(meshes) > 1 else meshes[0]
                merged.save(str(geometry_path))
            else:
                raise ValueError("This target renderer does not expose reusable scene geometry in the requested format.")
        if png_path is not None:
            png_path.parent.mkdir(parents=True, exist_ok=True)
            # ``screenshot()`` only works after the Plotter owns a render
            # window; ``show`` creates that renderer and captures the actual
            # scene without entering an interactive loop. Do this last because
            # auto-close tears down the renderer and its scene mesh registry.
            scene.show(screenshot=str(png_path), interactive=False, auto_close=True)
        return
    if geometry_path is not None:
        raise ValueError("Reusable scene geometry is unavailable for this target; printable SVG/PDF is not scene geometry.")
    import matplotlib.pyplot as plt
    if style:
        raise ValueError("This target renderer does not support managed 3D styles.")

    captured = []
    original_show = plt.show
    try:
        def capture_show(*_args, **_kwargs):
            captured.extend(plt.get_fignums())
        plt.show = capture_show
        target.plot()
        if not captured:
            captured = list(plt.get_fignums())
        if not captured:
            raise ValueError("The target renderer produced no Matplotlib figure to save.")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(captured[-1]).savefig(png_path, format="png", dpi=160)
    finally:
        plt.show = original_show


def _apply_target_style(scene, style: dict) -> None:
    """Apply supported PyVista cosmetics without changing target geometry."""
    if not style:
        return
    from pyCamSet.utils.visualisation import _apply_3d_cosmetics

    if style.get("legend", False):
        raise ValueError("Target scenes have no scalar legend to customise.")
    _apply_3d_cosmetics(
        scene, style.get("theme", "Light"), style.get("background", "theme"),
        style.get("point_size", 3.0), style.get("view", "isometric"),
        style.get("axes", True),
    )


if __name__ == "__main__":
    sys.exit(main())
