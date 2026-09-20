"""
Turning detections into calibrated cameras.

``calibrate_cameras`` is fetched on first use rather than imported here:
:mod:`zhang` and :mod:`telecentric` are reached from a target's own
``initial_calibration``, and importing this package eagerly would import
:mod:`camera_calibrator`, which imports the targets back.
"""

__all__ = ["calibrate_cameras"]


def __getattr__(name: str):
    if name == "calibrate_cameras":
        from .camera_calibrator import calibrate_cameras
        return calibrate_cameras
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
