"""PySide6 GUI package for the phased pyCamSet calibration workflow.

The package intentionally exposes no eager application object: importing
``pyCamSet.gui`` remains safe for headless tooling, while ``python -m
pyCamSet.gui`` owns QApplication creation and the event loop.
"""

__all__ = []
