from pathlib import Path


from pyCamSet.calibration_targets.ccube2 import target as cc2

_DEFAULT_OUTPUT_DIR = Path.cwd() / "calibration_targets" / "2D"
_DEFAULT_DICT_NAME = "DICT_4X4_1000"


def build_ccube2(
    n_points: int,
    length: float,
    aruco_dict: int | str = _DEFAULT_DICT_NAME,
    border_fraction: float = 0.1,
) -> cc2.Ccube2:
    """Instantiate a Ccube2 target from explicit cube parameters.

    There is no ``marker_backend``: a Ccube2's faces are read with aruco2 only.
    """
    return cc2.Ccube2(
        length,
        n_points,
        aruco_dict=aruco_dict,
        border_fraction=border_fraction,
    )


def default_output_name(n_points: int, length: float, export_kind: str) -> str:
    """The filename a target of this size is written to by default."""
    return cc2.Ccube2.printable_name({"n_points": n_points, "length": length}, export_kind)


def generate_ccube2_target(
    n_points: int = 5,
    length: float = 10,
    aruco_dict: int | str = _DEFAULT_DICT_NAME,
    border_fraction: float = 0.1,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    file_name: str | None = None,
    export_kind: str = "svg",
) -> tuple[cc2.Ccube2, Path]:
    """Create and save a Ccube2 target.

    Parameters
    ----------
    export_kind:
        One of ``"pdf_raster"``, ``"pdf_vector"``, or ``"svg"``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if file_name is None or not str(file_name).strip():
        file_name = default_output_name(n_points, length, export_kind)

    cube = build_ccube2(
        n_points=n_points,
        length=length,
        aruco_dict=aruco_dict,
        border_fraction=border_fraction,
    )
    out_path = out_dir / file_name

    saved_path = cube.save_printable(out_path, export_kind)

    return cube, saved_path


def main() -> None:
    """Write a default 5-square, 10 mm cube when run directly."""
    n_points = 5
    length = 10
    generate_ccube2_target(
        n_points=n_points,
        length=length,
        output_dir=_DEFAULT_OUTPUT_DIR,
        export_kind="svg",
    )


if __name__ == "__main__":
    main()
