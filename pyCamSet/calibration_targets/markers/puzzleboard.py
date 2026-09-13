from __future__ import annotations  # Keep annotations consistent with pyCamSet targets.

import cv2  # Convert pyCamSet's OpenCV images to the detector's expected colour order.
import numpy as np  # Type and shape normalisation for detector inputs.

from pyCamSet.calibration_targets.core.parameters import (
    Parameter,
    DetectorParameterisation,
)


def prepare_puzzleboard_image(image: np.ndarray) -> np.ndarray:
    """Convert a pyCamSet image to the RGB format expected by PuzzleBoard."""
    image = np.asarray(image)  # Accept array-like image inputs without copying unnecessarily.
    if image.ndim == 2:  # The detector accepts a three-channel image after this conversion.
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Expand grayscale input to RGB.
    if image.ndim != 3 or image.shape[2] not in (3, 4):  # Reject unsupported image layouts early.
        raise ValueError("PuzzleBoard images must be grayscale, BGR, RGB, BGRA, or RGBA arrays.")
    if image.shape[2] == 4:  # OpenCV images may include an alpha channel.
        image = image[:, :, :3]  # Discard alpha before changing colour order.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # pyCamSet loads images in BGR order.


def detect_puzzleboard_image(
    image: np.ndarray,
    min_width: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect PuzzleBoard grid points from a pyCamSet/OpenCV image."""
    # Imported here rather than at module scope: puzzle_board is an optional
    # extra, and a module-level import makes it a hard requirement of every
    # import that reaches this file -- the GUI among them.
    from puzzle_board.puzzle_board_detector import detect_puzzleboard  # Upstream PuzzleBoard repository.

    detector_image = prepare_puzzleboard_image(image)  # Normalise the image before external detection.
    return detect_puzzleboard(detector_image, min_width=int(min_width))  # Preserve the original detector output.


class PuzzleBoardDetector(DetectorParameterisation):
    """
    The PuzzleBoard repository's detector.

    ``min_width`` was a constructor argument of both PuzzleBoard targets,
    which made it look like part of their geometry: a target detected at a
    different ``min_width`` was refused as a different target, though the
    printed board and the meaning of every key it returns are identical.
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
                    "Concept: the smallest decoded grid the detector will "
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
