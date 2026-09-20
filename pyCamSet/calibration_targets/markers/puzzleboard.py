from __future__ import annotations

import cv2
import numpy as np

from pyCamSet.calibration_targets.core.parameters import (
    Parameter,
    DetectorParameterisation,
)


def prepare_puzzleboard_image(image: np.ndarray) -> np.ndarray:
    """Convert a pyCamSet image to the RGB format expected by PuzzleBoard."""
    image = np.asarray(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("PuzzleBoard images must be grayscale, BGR, RGB, BGRA, or RGBA arrays.")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_puzzleboard_image(
    image: np.ndarray,
    min_width: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect PuzzleBoard grid points from a pyCamSet/OpenCV image."""
    # puzzle_board is an optional extra: a module-level import would make
    # it a hard requirement of everything that reaches this file.
    from puzzle_board.puzzle_board_detector import detect_puzzleboard

    detector_image = prepare_puzzleboard_image(image)
    return detect_puzzleboard(detector_image, min_width=int(min_width))


class PuzzleBoardDetector(DetectorParameterisation):
    """
    The PuzzleBoard repository's detector.

    ``min_width`` belongs here rather than to a target's geometry: the same
    printed board read at a different ``min_width`` is the same target.
    """

    name = "puzzle_board"

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return (
            Parameter(
                key="min_width",
                label="Min Grid Width",
                default=4,
                dtype="int",
                tunable=True,
                settable=True,
                minimum=1,
                maximum=501,
                step=1,
                search_order=1,
                priority="A",
                concept=(
                    "the smallest decoded grid the detector will "
                    "accept. Detection: a recovered patch narrower than this "
                    "is discarded before its position is decoded. "
                    "Calibration: raising it drops small or oblique views of "
                    "the board, lowering it admits patches too small to "
                    "decode reliably and risks mislabelled keys."),
                range_text="At least 1; the printed board is at most 501 squares across",
                range_source="Estimated by us",
                suggested="4",
            ),
        )

    def unavailable_reason(self, values: dict | None = None) -> str | None:
        try:
            import puzzle_board  # noqa: F401
        except (ImportError, ModuleNotFoundError, OSError):
            return (
                "This target is read by the PuzzleBoard detector, which is "
                "not installed. Install the 'puzzle_board' package to detect "
                "with it.")
        return None


#: Shared rather than built per target: it describes nothing per instance.
PUZZLEBOARD_DETECTOR = PuzzleBoardDetector()
