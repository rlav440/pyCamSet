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
    min_width: int,
) -> tuple[int, int, float, int, int, float, float, int]:
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
        int(min_width),  # Normalise detector minimum width.
    )


def build_puzzleboard(
    num_squares_x: int | None = None,
    num_squares_y: int | None = None,
    square_size: float | None = None,
    start_x: int = 0,
    start_y: int = 0,
    paper_width: float = 210.0,
    paper_height: float = 297.0,
    min_width: int = 4,
) -> pb.PuzzleBoard:
    """Instantiate a PuzzleBoard target from explicit board parameters."""
    x, y, size, sx, sy, page_w, page_h, detector_width = _resolve_inputs(  # Resolve all public defaults once.
        num_squares_x,
        num_squares_y,
        square_size,
        start_x,
        start_y,
        paper_width,
        paper_height,
        min_width,
    )
    return pb.PuzzleBoard(  # Construct the target with the normalised parameters.
        num_squares_x=x,
        num_squares_y=y,
        square_size=size,
        start_x=sx,
        start_y=sy,
        paper_width=page_w,
        paper_height=page_h,
        min_width=detector_width,
    )


def default_output_name(
    num_squares_x: int,
    num_squares_y: int,
    square_size: float,
    export_kind: str,
) -> str:
    """Return a default output filename for the requested export kind."""
    suffix = ".svg" if export_kind == "svg" else ".pdf"  # Both PDF modes use the PDF extension.
    return (
        f"puzzleboard_{int(num_squares_x)}x{int(num_squares_y)}_"
        f"{float(square_size):g}mm{suffix}"
    )


def generate_puzzleboard_target(
    num_squares_x: int | None = None,
    num_squares_y: int | None = None,
    square_size: float | None = None,
    start_x: int = 0,
    start_y: int = 0,
    paper_width: float = 210.0,
    paper_height: float = 297.0,
    min_width: int = 4,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    file_name: str | None = None,
    export_kind: str = "svg",
) -> tuple[pb.PuzzleBoard, Path]:
    """Create and save a PuzzleBoard target."""
    x, y, size, sx, sy, page_w, page_h, detector_width = _resolve_inputs(  # Resolve parameters before naming files.
        num_squares_x,
        num_squares_y,
        square_size,
        start_x,
        start_y,
        paper_width,
        paper_height,
        min_width,
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
        min_width=detector_width,
    )
    out_path = out_dir / file_name  # Combine the destination directory and requested filename.
    if export_kind == "svg":  # Save a directly editable vector file.
        board.save_to_svg(out_path)
    elif export_kind == "pdf_vector":  # Save a PDF while preserving vector geometry.
        board.save_to_pdf(out_path, data_format="vector")
    elif export_kind == "pdf_raster":  # Save a compatibility raster PDF.
        board.save_to_pdf(out_path, data_format="raster")
    else:  # Reject unsupported export spellings explicitly.
        raise ValueError("export_kind must be one of: pdf_raster, pdf_vector, svg")
    return board, out_path.with_suffix("." + ("svg" if export_kind == "svg" else "pdf")).resolve()  # Return the actual saved path.


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

