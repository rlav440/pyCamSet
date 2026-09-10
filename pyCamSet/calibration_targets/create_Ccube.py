from pathlib import Path

import cv2

from pyCamSet.calibration_targets import target_Ccube as cc

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
        aruco_dict=_normalise_aruco_dict(aruco_dict, marker_backend),
        marker_backend=marker_backend,
    )


def default_output_name(n_points: int, length: float, export_kind: str) -> str:
    """Return a default output filename for the requested export kind."""
    suffix = ".svg" if export_kind == "svg" else ".pdf"
    return f"ccube_{int(n_points)}points_{float(length):g}mm{suffix}"


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

    if export_kind == "pdf_raster":
        saved_path = cube.save_to_pdf(out_path)
    elif export_kind == "pdf_vector":
        saved_path = cube.save_to_pdf(out_path, data_format="vector")
    elif export_kind == "svg":
        saved_path = cube.save_to_svg(out_path)
    else:
        raise ValueError("export_kind must be one of: pdf_raster, pdf_vector, svg")

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
