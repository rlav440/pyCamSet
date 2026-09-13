from pathlib import Path  # Keep output-path handling consistent with create_charuco.py.

from pyCamSet.calibration_targets import target_puzzleboard as pb  # Import the PuzzleBoard target implementation.

# Match the existing target-generation location.
_DEFAULT_OUTPUT_DIR = Path.cwd() / "calibration_targets" / "2D"


def _resolve_inputs(
    num_squares_x: int | None,
    num_squares_y: int | None,
    square_size: float | None,
    start_x: int,
    start_y: int,
    paper_width: float,
    paper_height: float,
) -> tuple[int, int, float, int, int, float, float]:
    """Apply defaults and normalise the public PuzzleBoard generator inputs."""
    if num_squares_x is None:  # Use the A4-width default from the 2 mm target workflow.
        num_squares_x = 105
    if num_squares_y is None:  # Use the A4-height default from the 2 mm target workflow.
        num_squares_y = 148
    if square_size is None:  # Use the requested fine-resolution square size by default.
        square_size = 2.0
    return (
        int(num_squares_x),  # Normalise horizontal square count.
        int(num_squares_y),  # Normalise vertical square count.
        float(square_size),  # Normalise physical square size in millimetres.
        int(start_x),  # Normalise horizontal code origin.
        int(start_y),  # Normalise vertical code origin.
        float(paper_width),  # Normalise page width in millimetres.
        float(paper_height),  # Normalise page height in millimetres.
    )


def build_puzzleboard(
    num_squares_x: int | None = None,
    num_squares_y: int | None = None,
    square_size: float | None = None,
    start_x: int = 0,
    start_y: int = 0,
    paper_width: float = 210.0,
    paper_height: float = 297.0,
) -> pb.PuzzleBoard:
    """Instantiate a PuzzleBoard target from explicit board parameters."""
    x, y, size, sx, sy, page_w, page_h = _resolve_inputs(  # Resolve all public defaults once.
        num_squares_x,
        num_squares_y,
        square_size,
        start_x,
        start_y,
        paper_width,
        paper_height,
    )
    return pb.PuzzleBoard(  # Construct the target with the normalised parameters.
        num_squares_x=x,
        num_squares_y=y,
        square_size=size,
        start_x=sx,
        start_y=sy,
        paper_width=page_w,
        paper_height=page_h,
    )


def default_output_name(num_squares_x: int,
    num_squares_y: int,
    square_size: float,
    export_kind: str,
) -> str:
    """The filename a target of this size is written to by default."""
    return pb.PuzzleBoard.printable_name(
        {"num_squares_x": num_squares_x, "num_squares_y": num_squares_y,
         "square_size": square_size}, export_kind)


def generate_puzzleboard_target(
    num_squares_x: int | None = None,
    num_squares_y: int | None = None,
    square_size: float | None = None,
    start_x: int = 0,
    start_y: int = 0,
    paper_width: float = 210.0,
    paper_height: float = 297.0,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    file_name: str | None = None,
    export_kind: str = "svg",
) -> tuple[pb.PuzzleBoard, Path]:
    """Create and save a PuzzleBoard target."""
    x, y, size, sx, sy, page_w, page_h = _resolve_inputs(  # Resolve parameters before naming files.
        num_squares_x,
        num_squares_y,
        square_size,
        start_x,
        start_y,
        paper_width,
        paper_height,
    )
    out_dir = Path(output_dir)  # Accept strings and Path objects like create_charuco.py.
    out_dir.mkdir(parents=True, exist_ok=True)  # Create the requested destination when needed.
    if file_name is None or not str(file_name).strip():  # Generate a descriptive filename by default.
        file_name = default_output_name(x, y, size, export_kind)
    board = build_puzzleboard(  # Build the target through the public factory.
        num_squares_x=x,
        num_squares_y=y,
        square_size=size,
        start_x=sx,
        start_y=sy,
        paper_width=page_w,
        paper_height=page_h,
    )
    out_path = out_dir / file_name  # Combine the destination directory and requested filename.
    return board, board.save_printable(out_path, export_kind)


def main() -> None:
    """Generate the default A4 PuzzleBoard SVG when this file is run directly."""
    num_squares_x = 105  # Fill the A4 width with 105 two-millimetre squares.
    num_squares_y = 148  # Fill 296 mm of the 297 mm A4 height.
    square_size = 2.0  # Keep the requested physical edge length.
    file_name = f"puzzleboard_{num_squares_x}x{num_squares_y}_{square_size:g}mm.svg"  # Name the vector output.
    generate_puzzleboard_target(  # Use the same direct-script style as create_charuco.py.
        num_squares_x=num_squares_x,
        num_squares_y=num_squares_y,
        square_size=square_size,
        output_dir=_DEFAULT_OUTPUT_DIR,
        file_name=file_name,
        export_kind="svg",
    )


if __name__ == "__main__":
    main()
