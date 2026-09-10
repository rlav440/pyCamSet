from __future__ import annotations  # Keep annotations consistent with pyCamSet targets.

import cv2  # Convert pyCamSet's OpenCV images to the detector's expected colour order.
import numpy as np  # Type and shape normalisation for detector inputs.
from puzzle_board.puzzle_board_detector import detect_puzzleboard  # Require and credit the upstream PuzzleBoard repository.


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
    detector_image = prepare_puzzleboard_image(image)  # Normalise the image before external detection.
    return detect_puzzleboard(detector_image, min_width=int(min_width))  # Preserve the original detector output.
