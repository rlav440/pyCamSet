from pathlib import Path


from pyCamSet.calibration_targets.charuco2 import target as ch2

_DEFAULT_OUTPUT_DIR = Path.cwd() / "calibration_targets" / "2D"
_DEFAULT_DICT_NAME = "DICT_4X4_1000"


def build_charuco2(
    num_squares_x: int = 5,
    num_squares_y: int = 5,
    square_size: float = 10.0,
    aruco_dict: int | str = _DEFAULT_DICT_NAME,
) -> ch2.ChArUco2:
    """Instantiate a ChArUco2 target from explicit board parameters.

    There is no ``marker_backend``: a ChArUco2 board is read with aruco2 only.
    """
    return ch2.ChArUco2(
        num_squares_x=int(num_squares_x),
        num_squares_y=int(num_squares_y),
        square_size=float(square_size),
        a_dict=aruco_dict,
    )


def default_output_name(num_squares_x: int,
    num_squares_y: int,
    square_size: float,
    export_kind: str,
) -> str:
    """The filename a target of this size is written to by default."""
    return ch2.ChArUco2.printable_name(
        {"num_squares_x": num_squares_x, "num_squares_y": num_squares_y,
         "square_size": square_size}, export_kind)


def generate_charuco2_target(
    num_squares_x: int = 5,
    num_squares_y: int = 5,
    square_size: float = 10.0,
    aruco_dict: int | str = _DEFAULT_DICT_NAME,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    file_name: str | None = None,
    export_kind: str = "svg",
) -> tuple[ch2.ChArUco2, Path]:
    """Create and save a ChArUco2 target.

    Parameters
    ----------
    export_kind:
        One of ``"pdf_raster"``, ``"pdf_vector"``, or ``"svg"``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if file_name is None or not str(file_name).strip():
        file_name = default_output_name(
            num_squares_x, num_squares_y, square_size, export_kind)

    board = build_charuco2(
        num_squares_x=num_squares_x,
        num_squares_y=num_squares_y,
        square_size=square_size,
        aruco_dict=aruco_dict,
    )
    out_path = out_dir / file_name

    return board, board.save_printable(out_path, export_kind)


def main() -> None:
    """Write a default 5x5, 10 mm board when run directly."""
    num_squares_x = 5
    num_squares_y = 5
    square_size = 10
    generate_charuco2_target(
        num_squares_x=num_squares_x,
        num_squares_y=num_squares_y,
        square_size=square_size,
        aruco_dict=_DEFAULT_DICT_NAME,
        output_dir=_DEFAULT_OUTPUT_DIR,
        export_kind="svg",
    )


if __name__ == "__main__":
    main()
