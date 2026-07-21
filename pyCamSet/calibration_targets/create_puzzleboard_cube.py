from pathlib import Path  # Keep output-path handling consistent with other target generators.

from pyCamSet.calibration_targets import target_puzzleboard_cube as pbc  # Import the PuzzleBoard cube target.

# Match the existing printable-target location.
_DEFAULT_OUTPUT_DIR = Path.cwd() / "calibration_targets" / "2D"


def build_puzzleboard_cube(
    num_squares_per_side: int = 20,
    square_size: float = 10.0,
    min_width: int = 4,
) -> pbc.PuzzleBoardCube:
    """Instantiate a deterministic PuzzleBoard cube from explicit parameters."""
    return pbc.PuzzleBoardCube(  # Construct the bounded six-face target.
        num_squares_per_side=int(num_squares_per_side),  # Normalise the face dimension.
        square_size=float(square_size),  # Normalise the physical square size in millimetres.
        min_width=int(min_width),  # Normalise the detector minimum width.
    )


def default_output_name(
    num_squares_per_side: int,
    square_size: float,
    export_kind: str,
) -> str:
    """Return a default output filename for the requested cube export kind."""
    suffix = ".svg" if export_kind == "svg" else ".pdf"  # Both PDF modes use a PDF suffix.
    return f"puzzleboard_cube_{int(num_squares_per_side)}x{int(num_squares_per_side)}_{float(square_size):g}mm{suffix}"


def generate_puzzleboard_cube_target(
    num_squares_per_side: int = 20,
    square_size: float = 10.0,
    min_width: int = 4,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    file_name: str | None = None,
    export_kind: str = "svg",
    border_width: float = 10.0,
    draw_cut_outline: bool = True,
    draw_face_ids: bool = True,
) -> tuple[pbc.PuzzleBoardCube, Path]:
    """Create and save a deterministic six-face PuzzleBoard cube net."""
    out_dir = Path(output_dir)  # Accept strings and Path objects like the other generators.
    out_dir.mkdir(parents=True, exist_ok=True)  # Create the output directory when needed.
    if file_name is None or not str(file_name).strip():  # Generate a descriptive filename by default.
        file_name = default_output_name(num_squares_per_side, square_size, export_kind)
    cube = build_puzzleboard_cube(  # Build the target through the public factory.
        num_squares_per_side=num_squares_per_side,
        square_size=square_size,
        min_width=min_width,
    )
    out_path = out_dir / file_name  # Combine the destination directory and requested filename.
    if export_kind == "svg":  # Save an editable vector net.
        cube.save_to_svg(
            out_path,
            border_width=border_width,
            draw_cut_outline=draw_cut_outline,
            draw_face_ids=draw_face_ids,
        )
    elif export_kind == "pdf_vector":  # Preserve SVG primitives in the PDF.
        cube.save_to_pdf(
            out_path,
            data_format="vector",
            border_width=border_width,
            draw_cut_outline=draw_cut_outline,
            draw_face_ids=draw_face_ids,
        )
    elif export_kind == "pdf_raster":  # Produce a compatibility raster PDF.
        cube.save_to_pdf(
            out_path,
            data_format="raster",
            border_width=border_width,
            draw_cut_outline=draw_cut_outline,
            draw_face_ids=draw_face_ids,
        )
    else:  # Reject unsupported export spellings explicitly.
        raise ValueError("export_kind must be one of: pdf_raster, pdf_vector, svg")
    return cube, out_path.with_suffix("." + ("svg" if export_kind == "svg" else "pdf")).resolve()  # Return the actual saved path.


def main() -> None:
    """Generate the default deterministic PuzzleBoard cube SVG when run directly."""
    num_squares_per_side = 20  # Use a conservative default face size for a printable cube net.
    square_size = 10.0  # Use a physically visible square size for the default target.
    file_name = default_output_name(num_squares_per_side, square_size, "svg")  # Name the vector net output.
    generate_puzzleboard_cube_target(  # Use the same direct-script style as the other target generators.
        num_squares_per_side=num_squares_per_side,
        square_size=square_size,
        output_dir=_DEFAULT_OUTPUT_DIR,
        file_name=file_name,
        export_kind="svg",
    )


if __name__ == "__main__":
    main()


