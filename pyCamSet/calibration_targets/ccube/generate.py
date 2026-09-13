from pathlib import Path


from pyCamSet.calibration_targets.ccube import target as cc
from pyCamSet.calibration_targets.markers.backend_registry import dictionary_id

_DEFAULT_OUTPUT_DIR = Path.cwd() / "calibration_targets" / "2D"
_DEFAULT_DICT_NAME = "DICT_4X4_1000"


def build_ccube(
    n_points: int,
    length: float,
    aruco_dict: int | str = _DEFAULT_DICT_NAME,
    marker_backend: str = "aruco1",
) -> cc.Ccube:
    """Instantiate a Ccube target using the same parameters as the legacy script."""
    return cc.Ccube(
        length,
        n_points,
        aruco_dict=dictionary_id(aruco_dict, marker_backend),
        marker_backend=marker_backend,
    )


def default_output_name(n_points: int, length: float, export_kind: str) -> str:
    """The filename a target of this size is written to by default."""
    return cc.Ccube.printable_name({"n_points": n_points, "length": length}, export_kind)


def generate_ccube_target(
    n_points: int = 5,
    length: float = 10,
    aruco_dict: int | str = _DEFAULT_DICT_NAME,
    marker_backend: str = "aruco1",
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

    cube = build_ccube(
        n_points=n_points,
        length=length,
        aruco_dict=aruco_dict,
        marker_backend=marker_backend,
    )
    out_path = out_dir / file_name

    saved_path = cube.save_printable(out_path, export_kind)

    return cube, saved_path


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
