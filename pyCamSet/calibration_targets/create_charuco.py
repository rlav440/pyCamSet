import warnings
from pathlib import Path

import cv2

from pyCamSet.calibration_targets import target_charuco as ch

_DEFAULT_OUTPUT_DIR = Path.cwd() / "calibration_targets" / "2D"
_DEFAULT_DICT_NAME = "DICT_4X4_1000"


def _normalise_aruco_dict(aruco_dict: int | str, marker_backend: str = "aruco1") -> int:
    if isinstance(aruco_dict, str):
        dict_name = aruco_dict.strip()
        if not dict_name:
            raise ValueError("aruco_dict cannot be empty.")
        if not dict_name.startswith("DICT_"):
            dict_name = f"DICT_{dict_name}"
        try:
            if marker_backend == "aruco2":
                import aruco2  # Lazy import: aruco2 is an optional dependency.
                return int(getattr(aruco2, dict_name))
            return int(getattr(cv2.aruco, dict_name))
        except AttributeError as exc:
            raise ValueError(f"Unknown ArUco dictionary name: {aruco_dict}") from exc
        except ImportError as exc:
            raise ImportError(
                "aruco2 is not installed. Install it with `pip install aruco2` "
                "to use marker_backend='aruco2'."
            ) from exc
    return int(aruco_dict)


def _resolve_legacy_inputs(
    num_squares_x: int | None,
    num_squares_y: int | None,
    square_size: float | None,
    n_points: int | None,
    length: float | None,
) -> tuple[int, int, float]:
    if n_points is not None:
        warnings.warn(
            "n_points is deprecated; use num_squares_x/num_squares_y.",
            DeprecationWarning,
            stacklevel=3,
        )
        if num_squares_x is None:
            num_squares_x = int(n_points)
        if num_squares_y is None:
            num_squares_y = int(n_points)

    if length is not None:
        warnings.warn(
            "length is deprecated; use square_size.",
            DeprecationWarning,
            stacklevel=3,
        )
        if square_size is None:
            square_size = float(length)

    if num_squares_x is None:
        num_squares_x = 5
    if num_squares_y is None:
        num_squares_y = 5
    if square_size is None:
        square_size = 10.0

    return int(num_squares_x), int(num_squares_y), float(square_size)


def build_charuco(
    num_squares_x: int | None = None,
    num_squares_y: int | None = None,
    square_size: float | None = None,
    marker_fraction: float = 0.8,
    aruco_dict: int | str = _DEFAULT_DICT_NAME,
    marker_backend: str = "aruco1",
    *,
    n_points: int | None = None,
    length: float | None = None,
) -> ch.ChArUco:
    """Instantiate a ChArUco target from explicit board parameters.

    Legacy aliases ``n_points`` and ``length`` are accepted for compatibility.
    """
    x, y, size = _resolve_legacy_inputs(num_squares_x, num_squares_y, square_size, n_points, length)
    return ch.ChArUco(
        num_squares_x=x,
        num_squares_y=y,
        square_size=size,
        marker_fraction=float(marker_fraction),
        a_dict=_normalise_aruco_dict(aruco_dict, marker_backend),
        marker_backend=marker_backend,
    )


def default_output_name(
    num_squares_x: int,
    num_squares_y: int,
    square_size: float,
    export_kind: str,
) -> str:
    """Return a default output filename for the requested export kind."""
    suffix = ".svg" if export_kind == "svg" else ".pdf"
    return (
        f"charuco_{int(num_squares_x)}x{int(num_squares_y)}_"
        f"{float(square_size):g}mm{suffix}"
    )


def generate_charuco_target(
    num_squares_x: int | None = None,
    num_squares_y: int | None = None,
    square_size: float | None = None,
    marker_fraction: float = 0.8,
    aruco_dict: int | str = _DEFAULT_DICT_NAME,
    marker_backend: str = "aruco1",
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    file_name: str | None = None,
    export_kind: str = "svg",
    *,
    n_points: int | None = None,
    length: float | None = None,
) -> tuple[ch.ChArUco, Path]:
    """Create and save a ChArUco target.

    Parameters
    ----------
    export_kind:
        One of ``"pdf_raster"``, ``"pdf_vector"``, or ``"svg"``.
    """
    x, y, size = _resolve_legacy_inputs(num_squares_x, num_squares_y, square_size, n_points, length)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if file_name is None or not str(file_name).strip():
        file_name = default_output_name(x, y, size, export_kind)

    board = build_charuco(
        num_squares_x=x,
        num_squares_y=y,
        square_size=size,
        marker_fraction=marker_fraction,
        aruco_dict=aruco_dict,
        marker_backend=marker_backend,
    )
    out_path = out_dir / file_name

    if export_kind == "pdf_raster":
        board.save_to_pdf(out_path)
    elif export_kind == "pdf_vector":
        board.save_to_pdf(out_path, data_format="vector")
    elif export_kind == "svg":
        board.save_to_svg(out_path)
    else:
        raise ValueError("export_kind must be one of: pdf_raster, pdf_vector, svg")

    return board, out_path


def main() -> None:
    """Preserve the original script defaults when run directly."""
    num_squares_x = 5
    num_squares_y = 5
    square_size = 10
    file_name = f"charuco_{num_squares_x}x{num_squares_y}_{square_size}mm.pdf"
    generate_charuco_target(
        num_squares_x=num_squares_x,
        num_squares_y=num_squares_y,
        square_size=square_size,
        marker_fraction=0.8,
        aruco_dict=_DEFAULT_DICT_NAME,
        output_dir=_DEFAULT_OUTPUT_DIR,
        file_name=file_name,
        export_kind="svg",
    )


if __name__ == "__main__":
    main()

