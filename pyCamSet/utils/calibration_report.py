"""
A structured summary of what a calibration run produced.

The optimisation leaves a residual vector and a parameter array on the camera
set, which is everything needed to judge a calibration and nothing anyone can
read. :class:`CalibrationReport` turns that into the numbers a person actually
wants -- the error distribution, the same distribution per camera, which
images are worst, what the solver did -- computed once at the end of the
bundle adjustment and carried on the camera set from there.

It is a plain dataclass so it can be compared between runs, serialised into
the ``.camset`` file next to the parameters, and printed.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from pyCamSet.utils import report_format as fmt

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

logger = logging.getLogger(__name__)

#: Above this initial mean reprojection error the starting point is suspect
#: rather than merely rough: usually a camera or the target is misplaced.
HIGH_INITIAL_ERROR_PX = 100.0

#: Above this final mean reprojection error the calibration has not really
#: converged onto the data and the result should not be trusted.
HIGH_FINAL_ERROR_PX = 5.0

#: How many of the worst images to name in the summary.
N_WORST_IMAGES = 5


def _distribution(values: np.ndarray) -> dict[str, float]:
    """
    The summary statistics used everywhere in the report.

    :param values: per point Euclidean reprojection errors, in pixels
    """
    if values.size == 0:
        return {k: float("nan") for k in
                ("mean_px", "median_px", "rms_px", "p95_px", "max_px")}
    return {
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "rms_px": float(np.sqrt(np.mean(values ** 2))),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(np.max(values)),
    }


@dataclass
class CameraErrorStats:
    """
    The reprojection error distribution for a single camera.
    """
    name: str
    index: int
    n_points: int
    mean_px: float
    median_px: float
    rms_px: float
    p95_px: float
    max_px: float


@dataclass
class ImageErrorStats:
    """
    The reprojection error distribution for a single image.
    """
    index: int
    n_points: int
    mean_px: float
    max_px: float


@dataclass
class CalibrationReport:
    """
    What a calibration run produced, as data rather than as log lines.

    :param camera_names: the cameras, in detection index order
    :param n_images: how many images the detection spans
    :param n_control_points: detected features that entered the optimisation
    :param n_parameters: free parameters the solver moved
    :param n_missing_poses: images with no usable target pose
    :param initial_error_px: mean Euclidean error before optimising
    :param per_camera: the error distribution for each camera
    :param worst_images: the worst images by mean error, worst first
    :param solver: which solver ran, 'schur' or 'trf'
    :param flags: the concerns worth a person's attention
    """
    camera_names: list[str]
    n_images: int
    n_control_points: int
    n_parameters: int
    n_missing_poses: int

    initial_error_px: float
    mean_px: float
    median_px: float
    rms_px: float
    p95_px: float
    max_px: float

    per_camera: list[CameraErrorStats] = field(default_factory=list)
    worst_images: list[ImageErrorStats] = field(default_factory=list)

    solver: str = ""
    solver_status: int | None = None
    solver_message: str = ""
    n_iterations: int | None = None
    n_function_evals: int | None = None
    duration_s: float = float("nan")

    flags: list[str] = field(default_factory=list)
    save_path: str | None = None

    @property
    def n_cameras(self) -> int:
        return len(self.camera_names)

    @classmethod
    def from_optimisation(
        cls,
        optimisation: OptimizeResult,
        param_handler,
        initial_error_px: float,
        duration_s: float,
        solver: str,
    ) -> CalibrationReport:
        """
        Build the report from a finished bundle adjustment.

        :param optimisation: the scipy style result the solver returned
        :param param_handler: the handler that defined the problem
        :param initial_error_px: mean Euclidean error before optimising
        :param duration_s: wall clock seconds the solve took
        :param solver: which solver ran, 'schur' or 'trf'
        """
        residuals = np.reshape(np.asarray(optimisation.fun, dtype=float), (-1, 2))
        euclid = np.linalg.norm(residuals, axis=1)

        detection = param_handler.detection
        overall = _distribution(euclid)

        report = cls(
            camera_names=list(detection.cam_names),
            n_images=int(detection.max_ims),
            n_control_points=int(euclid.size),
            n_parameters=int(np.size(optimisation.x)),
            n_missing_poses=(
                int(np.sum(param_handler.missing_poses))
                if getattr(param_handler, "missing_poses", None) is not None
                else 0
            ),
            initial_error_px=float(initial_error_px),
            solver=solver,
            solver_status=_maybe_int(getattr(optimisation, "status", None)),
            solver_message=str(getattr(optimisation, "message", "")),
            n_iterations=_maybe_int(getattr(optimisation, "nit", None)),
            n_function_evals=_maybe_int(getattr(optimisation, "nfev", None)),
            duration_s=float(duration_s),
            **overall,
        )

        report.per_camera, report.worst_images = _per_group_stats(
            euclid, detection)
        report.flags = report._find_flags()
        return report

    def _find_flags(self) -> list[str]:
        """
        The concerns worth raising, each naming its threshold and its value.
        """
        flags = []
        if np.isnan(self.mean_px):
            flags.append(
                "the final error is NaN: the optimisation did not produce a "
                "usable result")
        elif self.mean_px > HIGH_FINAL_ERROR_PX:
            flags.append(
                f"final mean error {self.mean_px:.2f} px is above the "
                f"{HIGH_FINAL_ERROR_PX:.0f} px this check expects; treat the "
                f"result as unconverged and check the worst cameras below")
        if self.initial_error_px > HIGH_INITIAL_ERROR_PX:
            flags.append(
                f"initial error {self.initial_error_px:.2f} px was above the "
                f"{HIGH_INITIAL_ERROR_PX:.0f} px this check expects, which "
                f"usually means a camera or the target started misplaced")
        if self.n_missing_poses:
            flags.append(
                f"{self.n_missing_poses} of {self.n_images} images had no "
                f"usable target pose and did not constrain the solve")
        return flags

    def to_dict(self) -> dict[str, Any]:
        """
        The report as plain types, for the ``.camset`` file or a diff.
        """
        out = asdict(self)
        out["n_cameras"] = self.n_cameras
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationReport:
        """
        Rebuild a report saved by :meth:`to_dict`.

        :param data: the dictionary to read
        """
        data = dict(data)
        data.pop("n_cameras", None)  # derived from camera_names
        data["per_camera"] = [CameraErrorStats(**c)
                              for c in data.get("per_camera", [])]
        data["worst_images"] = [ImageErrorStats(**i)
                                for i in data.get("worst_images", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def summary(self, colour: bool | None = None) -> str:
        """
        The report as a block of text that fits an 80 column terminal.

        The per camera errors are graded by colour: blue under 0.1 px for an
        exceptional calibration, green under 1 px, orange under 5 px, and red
        above that, which is the same 5 px the high error flag uses.

        :param colour: force colour on or off, defaulting to auto detection
        """
        duration = ("  n/a" if np.isnan(self.duration_s)
                    else f"{self.duration_s:.1f} s")

        lines = fmt.title("Calibration summary")
        lines += fmt.kv_rows([
            ("cameras", str(self.n_cameras)),
            ("control points", str(self.n_control_points)),
            ("images", str(self.n_images)),
            ("free parameters", str(self.n_parameters)),
            ("solver", self.solver or "n/a"),
            ("iterations", _or_na(self.n_iterations)),
            ("missing poses", str(self.n_missing_poses)),
            ("duration", duration),
        ])
        lines += [
            "",
            "  Reprojection error, px",
            f"    initial {self.initial_error_px:8.2f}   ->"
            f"   final mean {self.mean_px:8.2f}",
            f"    median {self.median_px:7.2f}    rms {self.rms_px:7.2f}"
            f"    p95 {self.p95_px:7.2f}    max {self.max_px:7.2f}",
        ]

        if self.per_camera:
            lines += [""]
            lines += fmt.table(
                ["camera", "points", "mean", "median", "p95", "max"],
                [[c.name, c.n_points, fmt.error_cell(c.mean_px),
                  fmt.error_cell(c.median_px), fmt.error_cell(c.p95_px),
                  fmt.error_cell(c.max_px)]
                 for c in self.per_camera],
                widths=[16, 8, 9, 9, 9, 9],
                colour=colour,
            )

        if self.worst_images:
            lines += [""] + fmt.wrap_items(
                "  worst images: ",
                [f"{i.index} ({i.mean_px:.2f} px)" for i in self.worst_images],
            )

        if self.solver_message:
            lines.append(f"  termination:  {self.solver_message}")

        lines += fmt.flag_lines(self.flags)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _per_group_stats(
    euclid: np.ndarray, detection,
) -> tuple[list[CameraErrorStats], list[ImageErrorStats]]:
    """
    Split the per point errors by camera and by image.

    The residuals come out of the loss function in the row order of the
    detection the loss was built from, two per detected feature, so the
    detection's own camera and image columns say which error belongs to
    what. A handler that builds its loss from something else would break
    that correspondence, so the lengths are checked rather than assumed.

    :param euclid: per point Euclidean errors, in pixels
    :param detection: the detection the loss function was built from
    :return: the per camera and the worst per image statistics
    """
    data = detection.get_data() if detection.has_data() else None
    if data is None or len(data) != euclid.size:
        logger.debug(
            "Skipping the per camera error breakdown: the detection has "
            f"{0 if data is None else len(data)} rows but the optimisation "
            f"returned {euclid.size} residual pairs.")
        return [], []

    cam_index = data[:, 0].astype(int)
    im_index = data[:, 1].astype(int)

    per_camera = []
    for index, name in enumerate(detection.cam_names):
        mask = cam_index == index
        if not np.any(mask):
            continue
        per_camera.append(CameraErrorStats(
            name=name, index=index, n_points=int(np.count_nonzero(mask)),
            **_distribution(euclid[mask]),
        ))

    per_image = []
    for index in np.unique(im_index):
        mask = im_index == index
        values = euclid[mask]
        per_image.append(ImageErrorStats(
            index=int(index), n_points=int(values.size),
            mean_px=float(np.mean(values)), max_px=float(np.max(values)),
        ))
    per_image.sort(key=lambda i: i.mean_px, reverse=True)

    return per_camera, per_image[:N_WORST_IMAGES]


def _maybe_int(value) -> int | None:
    """
    An int when there is one, None otherwise: solvers disagree on which of
    these fields they set.

    :param value: the value to coerce
    """
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _or_na(value) -> str:
    return "n/a" if value is None else str(value)


