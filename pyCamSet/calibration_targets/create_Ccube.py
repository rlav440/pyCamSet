"""
Purpose: Generate a raster PDF, vector PDF, or SVG of a Ccube target.

Note: These conversions were created as, with higher resolution cameras,
rasters create artifacts that reduce the accuracy of corner and marker detection.
"""

from pathlib import Path
from pyCamSet.calibration_targets import target_Ccube as cc

_DEFAULT_OUTPUT_DIR = Path(r"D:\Work\calibration_targets\2D")


def build_ccube(n_points: int, length: float) -> cc.Ccube:
    """Instantiate a Ccube target using the same parameters as the legacy script."""
    return cc.Ccube(length, n_points)


def default_output_name(n_points: int, length: float, export_kind: str) -> str:
    """Return a default output filename for the requested export kind."""
    suffix = ".svg" if export_kind == "svg" else ".pdf"
    return f"ccube_{int(n_points)}points_{float(length):g}mm{suffix}"


def generate_ccube_target(
    n_points: int = 5,
    length: float = 10,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    file_name: str | None = None,
    export_kind: str = "svg",
) -> tuple[cc.Ccube, Path]:
    """Create and save a Ccube target.

    Parameters
    ----------
    export_kind:
        One of ``"pdf_raster"``, ``"pdf_vector"``, or ``"svg"``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if file_name is None or not str(file_name).strip():
        file_name = default_output_name(n_points, length, export_kind)

    cube = build_ccube(n_points=n_points, length=length)
    out_path = out_dir / file_name

    if export_kind == "pdf_raster":
        cube.save_to_pdf(out_path)
    elif export_kind == "pdf_vector":
        cube.save_to_pdf(out_path, data_format="vector")
    elif export_kind == "svg":
        cube.save_to_svg(out_path)
    else:
        raise ValueError("export_kind must be one of: pdf_raster, pdf_vector, svg")

    return cube, out_path


def main() -> None:
    """Preserve the original script defaults when run directly."""
    n_points = 5
    length = 10
    file_name = f"ccube_{n_points}points_{length}mm.pdf"
    generate_ccube_target(
        n_points=n_points,
        length=length,
        output_dir=_DEFAULT_OUTPUT_DIR,
        file_name=file_name,
        export_kind="svg",
    )


if __name__ == "__main__":
    main()
