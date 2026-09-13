from pathlib import Path  # Keep output-path handling consistent with other target generators.

from pyCamSet.calibration_targets.puzzleboard_cube import target as pbc  # Import the PuzzleBoard cube target.

# Match the existing printable-target location.
_DEFAULT_OUTPUT_DIR = Path.cwd() / "calibration_targets" / "2D"


def build_puzzleboard_cube(
    n_points: int = 20,
    length: float = 200.0,
) -> pbc.PuzzleBoardCube:
    """Instantiate a deterministic PuzzleBoard cube from explicit parameters."""
    return pbc.PuzzleBoardCube(  # Construct the bounded six-face target.
        n_points=int(n_points),  # Normalise the face dimension.
        length=float(length),  # Normalise the physical square size in millimetres.
    )


def default_output_name(n_points: int,
    length: float,
    export_kind: str,
) -> str:
    """The filename a target of this size is written to by default."""
    return pbc.PuzzleBoardCube.printable_name(
        {"n_points": n_points, "length": length}, export_kind)


def generate_puzzleboard_cube_target(
    n_points: int = 20,
    length: float = 200.0,
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
        file_name = default_output_name(n_points, length, export_kind)
    cube = build_puzzleboard_cube(  # Build the target through the public factory.
        n_points=n_points,
        length=length,
    )
    out_path = out_dir / file_name  # Combine the destination directory and requested filename.
    return cube, cube.save_printable(
        out_path,
        export_kind,
        border_width=border_width,
        draw_cut_outline=draw_cut_outline,
        draw_face_ids=draw_face_ids,
    )


def main() -> None:
    """Generate the default deterministic PuzzleBoard cube SVG when run directly."""
    n_points = 20  # Use a conservative default face size for a printable cube net.
    length = 200.0  # Use a physically visible square size for the default target.
    file_name = default_output_name(n_points, length, "svg")  # Name the vector net output.
    generate_puzzleboard_cube_target(  # Use the same direct-script style as the other target generators.
        n_points=n_points,
        length=length,
        output_dir=_DEFAULT_OUTPUT_DIR,
        file_name=file_name,
        export_kind="svg",
    )


if __name__ == "__main__":
    main()
