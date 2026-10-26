'''Exercise each supported target/backend row through Phase 1 detection.
'''
from __future__ import annotations

from pathlib import Path

import pytest

from tools.edu_release.release_matrix import CORPUS_ROWS, ROWS


@pytest.mark.parametrize("row", ROWS, ids=lambda row: f"{row}-p1")
def test_release_cell(row, record_property, tmp_path):
    """Detect the checked-in corpus rows or a single rendered target face."""
    # The target's find_in_image method is the per-image detector Phase 1 calls
    # from AbstractTarget.find_in_imfolder; synthetic cells exercise that API.
    import cv2
    import numpy as np

    from pyCamSet import calibration_targets
    from pyCamSet.calibration_targets.markers.backend_registry import (
        marker_backend_available,
    )

    family, backend = row.split("-", 1)
    if backend in {"aruco2", "puzzle_board"} and not marker_backend_available(
            "aruco2" if backend == "aruco2" else "aruco1"):
        pytest.skip(f"untested: optional {backend} backend is unavailable")
    if backend == "puzzle_board":
        try:
            import puzzle_board  # noqa: F401
        except (ImportError, OSError) as error:
            pytest.skip(f"untested: puzzle_board is unavailable ({error})")
    if family in {"CIco", "CIco2", "PuzzleBoardIco", "PuzzleBoard", "PuzzleBoardCube"}:
        try:
            # Use the drawing API's loader before declaring Cairo unavailable:
            # a headless Windows process need not inherit an activated conda PATH.
            # Source: pyCamSet/utils/cairo_dll_helper.py; calibration_targets/puzzleboard.py
            from pyCamSet.utils.cairo_dll_helper import cairosvg_or_explain

            cairosvg = cairosvg_or_explain()
        except (ImportError, OSError) as error:
            pytest.skip(f"untested: native cairo is unavailable ({error})")

    if row in CORPUS_ROWS:
        corpus = Path(__file__).parent / "test_data"
        if row == "ChArUco-aruco1":
            target = calibration_targets.ChArUco(20, 20, 4)
            image_path = corpus / "calibration_charuco" / "1" / "cal1.jpg"
        else:
            target = calibration_targets.Ccube(
                n_points=10, length=40, aruco_dict=cv2.aruco.DICT_6X6_1000,
                border_fraction=0.2,
            )
            image_path = corpus / "calibration_ccube" / "cam0" / "0.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        assert image is not None, f"corpus image could not be read: {image_path}"
        record_property("coverage", "image-corpus")
    else:
        if backend == "aruco2":
            try:
                import aruco2  # noqa: F401
            except (ImportError, OSError) as error:
                pytest.skip(f"untested: aruco2 is unavailable ({error})")

        if family == "ChArUco":
            target = calibration_targets.ChArUco(
                7, 6, 10.0, marker_backend=backend)
            image = target._render_board(px_per_mm=12.0)
        elif family == "ChArUco2":
            from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
                render_grid_board_image,
            )

            target = calibration_targets.ChArUco2(7, 6, 10.0)
            image = render_grid_board_image(
                target.grid_size, target._aruco_dict_int, 18)
        elif family == "Ccube":
            target = calibration_targets.Ccube(
                n_points=5, length=40.0, marker_backend=backend)
            image = np.ascontiguousarray(target.textures[0], dtype=np.uint8)
        elif family == "Ccube2":
            target = calibration_targets.Ccube2(n_points=5, length=40.0)
            image = target.face_texture(0)
        elif family == "CIco":
            target = calibration_targets.CIco(
                length=100.0, n_points=10, marker_backend=backend)
            image = cv2.cvtColor(
                target._face_texture(0, 1400), cv2.COLOR_RGB2GRAY)
        elif family == "CIco2":
            target = calibration_targets.CIco2(length=100.0, n_points=12)
            image = cv2.cvtColor(
                target._face_texture(0, 1600), cv2.COLOR_RGB2GRAY)
        elif family == "PuzzleBoard":
            from io import BytesIO

            from PIL import Image

            target = calibration_targets.PuzzleBoard(
                num_squares_x=12, num_squares_y=12, square_size=10.0,
                paper_width=140.0, paper_height=140.0)
            png = cairosvg.svg2png(
                bytestring=target._svg_document().tostring().encode("utf-8"),
                output_width=1200)
            with Image.open(BytesIO(png)) as rendered:
                image = np.asarray(rendered.convert("L"))
        elif family == "PuzzleBoardCube":
            # Use the first printed face's exact SVG geometry, clipped from the
            # target's generated net, matching the upstream face-render tests.
            from io import BytesIO

            from PIL import Image

            target = calibration_targets.PuzzleBoardCube(n_points=5, length=40.0)
            png = cairosvg.svg2png(
                bytestring=target._face_svg(0).encode("utf-8"),
                output_width=1600,
            )
            with Image.open(BytesIO(png)) as rendered:
                image = np.asarray(rendered.convert("L"))
        elif family == "PuzzleBoardIco":
            from pyCamSet.calibration_targets.puzzleboard_ico import PuzzleBoardIco

            target = PuzzleBoardIco(length=100.0, n_points=16)
            image = cv2.cvtColor(
                target._face_texture(0, 1600), cv2.COLOR_RGB2GRAY)
        else:  # A newly added ROW must get an explicit render path.
            pytest.skip(f"untested: no Phase 1 render path for {row}")
        record_property("coverage", "synthetic-render")

    detection = target.find_in_image(image)
    assert detection is not None and detection.has_data, f"{row} found no target points"
