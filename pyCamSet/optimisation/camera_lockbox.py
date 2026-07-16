'''
Purpose: Build Gaussian camera lockbox priors for pyCamSet bundle adjustment.
Status:  Backend MVP for Phase 3 extrinsic-parameter lockboxes.
Future:  Add deliberately designed world-position lockboxes and Phase 4 handling.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import sparse


@dataclass(slots=True)
class CameraLockboxConfig:
    """Configuration for hard extrinsic bounds plus soft Gaussian priors."""

    enabled: bool = False
    rotation_half_width: float = 0.1
    translation_half_width: float = 0.1
    rotation_sigma: float = 0.05
    translation_sigma: float = 0.01
    # World-space center prior: constrains C = -R^T @ t directly.
    # 0.0 = disabled; any positive value enables the prior with that sigma.
    center_sigma: float = 0.0


@dataclass(slots=True)
class CameraLockboxPrior:
    """Packed-parameter lockbox prior data.

    ``indices`` are exact optimisation-vector indices. They are not raw camera
    indices, and must not be multiplied by six by residual or Jacobian code.
    """

    enabled: bool
    param_len: int
    indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    centres: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sigmas: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    lower_bounds: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    upper_bounds: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    camera_names: tuple[str, ...] = ()
    implementation_mode: str = "extrinsic_parameter_mvp"
    # World-space center prior data — one row per constrained camera.
    world_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    cam_base_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    center_sigma: float = 0.0


# ── Rodrigues helpers ────────────────────────────────────────────────────────


def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Pure-numpy Rodrigues vector → 3×3 rotation matrix."""
    r = np.asarray(rvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(r))
    # Skew-symmetric matrix of r (not unit k yet).
    K = np.array([[0.0, -r[2], r[1]],
                  [r[2],  0.0, -r[0]],
                  [-r[1], r[0],  0.0]])
    if theta < 1e-9:
        return np.eye(3) + K  # first-order approximation for near-zero rotation
    k = r / theta
    # R = cos(θ)I + (1−cos(θ))kkᵀ + sin(θ)[k]×
    return (np.cos(theta) * np.eye(3)
            + (1.0 - np.cos(theta)) * np.outer(k, k)
            + np.sin(theta) * (K / theta))


def _camera_center(params: np.ndarray, base: int) -> np.ndarray:
    """World-space camera center C = −R^T @ t from packed params at offset base."""
    R = _rodrigues_to_matrix(params[base:base + 3])
    return -(R.T @ params[base + 3:base + 6])


def _center_jac_block(params: np.ndarray, base: int, eps: float = 1e-7) -> np.ndarray:
    """Numerical 3×6 Jacobian of C = −R^T @ t w.r.t. params[base : base+6]."""
    C0 = _camera_center(params, base)
    p = params.copy()
    jac = np.zeros((3, 6), dtype=float)
    for j in range(6):
        p[base + j] += eps
        jac[:, j] = (_camera_center(p, base) - C0) / eps
        p[base + j] = params[base + j]
    return jac


# ── Validation ────────────────────────────────────────────────────────────────


def _validate_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Camera lockbox {name} must be finite and positive; got {value!r}")


# ── Prior construction ────────────────────────────────────────────────────────


def make_disabled_prior(param_len: int) -> CameraLockboxPrior:
    """Return an inert prior with no constrained entries."""
    return CameraLockboxPrior(
        enabled=False,
        param_len=int(param_len),
        lower_bounds=np.full(int(param_len), -np.inf, dtype=float),
        upper_bounds=np.full(int(param_len), np.inf, dtype=float),
    )


def build_extrinsic_parameter_lockbox(
    cam_names: Sequence[str],
    extr_unfixed: np.ndarray,
    source_extrinsics: np.ndarray | None,
    intr_end: int,
    param_len: int,
    config: CameraLockboxConfig,
) -> CameraLockboxPrior:
    """Build a lockbox over packed Rodrigues+translation extrinsic entries."""
    param_len = int(param_len)
    if not config.enabled:
        return make_disabled_prior(param_len)

    if source_extrinsics is None:
        raise ValueError("Camera lockbox is enabled, but no source extrinsics were supplied")

    source_extrinsics = np.asarray(source_extrinsics, dtype=float)
    extr_unfixed = np.asarray(extr_unfixed, dtype=bool)
    cam_names = tuple(str(name) for name in cam_names)

    if source_extrinsics.shape != (len(cam_names), 6):
        raise ValueError(
            "Camera lockbox source_extrinsics must have shape "
            f"({len(cam_names)}, 6); got {source_extrinsics.shape}"
        )
    if extr_unfixed.shape[0] != len(cam_names):
        raise ValueError(
            "Camera lockbox extr_unfixed length must match cam_names; "
            f"got {extr_unfixed.shape[0]} and {len(cam_names)}"
        )

    _validate_positive("rotation_half_width", config.rotation_half_width)
    _validate_positive("translation_half_width", config.translation_half_width)
    _validate_positive("rotation_sigma", config.rotation_sigma)
    _validate_positive("translation_sigma", config.translation_sigma)
    if config.center_sigma < 0:
        raise ValueError(f"Camera lockbox center_sigma must be non-negative; got {config.center_sigma!r}")

    lower_bounds = np.full(param_len, -np.inf, dtype=float)
    upper_bounds = np.full(param_len, np.inf, dtype=float)
    constrained_indices: list[int] = []
    centres: list[float] = []
    sigmas: list[float] = []
    constrained_names: list[str] = []
    world_centers: list[list[float]] = []
    cam_base_idx: list[int] = []

    packed_extrinsic_slot = 0
    for raw_camera_index, is_unfixed in enumerate(extr_unfixed):
        if not is_unfixed:
            continue

        base_index = int(intr_end) + packed_extrinsic_slot * 6
        packed_extrinsic_slot += 1
        if base_index + 6 > param_len:
            raise ValueError(
                "Camera lockbox packed extrinsic index exceeds parameter vector: "
                f"camera={cam_names[raw_camera_index]!r}, base={base_index}, param_len={param_len}"
            )

        camera_centre = source_extrinsics[raw_camera_index]
        camera_sigmas = np.array(
            [config.rotation_sigma] * 3 + [config.translation_sigma] * 3,
            dtype=float,
        )
        half_widths = np.array(
            [config.rotation_half_width] * 3 + [config.translation_half_width] * 3,
            dtype=float,
        )
        camera_indices = np.arange(base_index, base_index + 6, dtype=int)

        lower_bounds[camera_indices] = camera_centre - half_widths
        upper_bounds[camera_indices] = camera_centre + half_widths
        constrained_indices.extend(int(idx) for idx in camera_indices)
        centres.extend(float(value) for value in camera_centre)
        sigmas.extend(float(value) for value in camera_sigmas)
        constrained_names.extend([cam_names[raw_camera_index]] * 6)

        # World-space center for this camera.
        C = _camera_center(camera_centre, 0)
        world_centers.append(C.tolist())
        cam_base_idx.append(base_index)

    return CameraLockboxPrior(
        enabled=True,
        param_len=param_len,
        indices=np.asarray(constrained_indices, dtype=int),
        centres=np.asarray(centres, dtype=float),
        sigmas=np.asarray(sigmas, dtype=float),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        camera_names=tuple(constrained_names),
        world_centers=np.asarray(world_centers, dtype=float).reshape(-1, 3),
        cam_base_indices=np.asarray(cam_base_idx, dtype=int),
        center_sigma=float(config.center_sigma),
    )


# ── Bound and residual helpers ────────────────────────────────────────────────


def apply_lockbox_bounds(param_len: int, prior: CameraLockboxPrior) -> tuple[np.ndarray, np.ndarray]:
    """Return SciPy least_squares-compatible lower and upper bound arrays."""
    param_len = int(param_len)
    if prior is None or not prior.enabled:
        return np.full(param_len, -np.inf, dtype=float), np.full(param_len, np.inf, dtype=float)
    if prior.param_len != param_len:
        raise ValueError(f"Camera lockbox prior length {prior.param_len} does not match {param_len}")
    return prior.lower_bounds.copy(), prior.upper_bounds.copy()


def append_lockbox_residuals(
    base_residuals: np.ndarray,
    params: np.ndarray,
    prior: CameraLockboxPrior,
) -> np.ndarray:
    """Append MAP residuals: one per constrained packed parameter, plus 3 per camera for world-center prior."""
    if prior is None or not prior.enabled:
        return base_residuals
    result = np.asarray(base_residuals)
    if prior.indices.size > 0:
        param_residuals = (np.asarray(params)[prior.indices] - prior.centres) / prior.sigmas
        result = np.concatenate((result, param_residuals))
    if prior.center_sigma > 0.0 and prior.world_centers.shape[0] > 0:
        p = np.asarray(params)
        parts = [
            (_camera_center(p, int(b)) - prior.world_centers[i]) / prior.center_sigma
            for i, b in enumerate(prior.cam_base_indices)
        ]
        result = np.concatenate((result, np.concatenate(parts)))
    return result


def append_lockbox_jacobian(
    base_jacobian,
    param_len: int,
    prior: CameraLockboxPrior,
    params: np.ndarray | None = None,
):
    """Append prior rows to a projection Jacobian.

    Existing rows: diagonal 1/sigma per constrained packed parameter.
    New rows (when center_sigma > 0 and params supplied): numerical 3×6
    Jacobian of C = −R^T @ t per camera, divided by center_sigma.
    """
    if prior is None or not prior.enabled:
        return base_jacobian

    result = base_jacobian

    # Diagonal rows for the packed-parameter prior (existing behaviour).
    if prior.indices.size > 0:
        row_indices = np.arange(prior.indices.size, dtype=int)
        param_rows = sparse.csr_array(
            (1.0 / prior.sigmas, (row_indices, prior.indices)),
            shape=(prior.indices.size, int(param_len)),
        )
        if sparse.issparse(result):
            result = sparse.vstack((result, param_rows), format="csr")
        else:
            result = np.vstack((np.asarray(result), param_rows.toarray()))

    # Nonlinear rows for the world-space center prior (new).
    if prior.center_sigma > 0.0 and prior.world_centers.shape[0] > 0 and params is not None:
        p = np.asarray(params)
        n_cam = prior.world_centers.shape[0]
        coo_rows: list[int] = []
        coo_cols: list[int] = []
        coo_data: list[float] = []
        for i, base in enumerate(prior.cam_base_indices):
            jac_block = _center_jac_block(p, int(base)) / prior.center_sigma  # (3, 6)
            row_offset = i * 3
            for r in range(3):
                for c in range(6):
                    v = float(jac_block[r, c])
                    if v != 0.0:
                        coo_rows.append(row_offset + r)
                        coo_cols.append(int(base) + c)
                        coo_data.append(v)
        center_rows = sparse.csr_array(
            (coo_data, (coo_rows, coo_cols)),
            shape=(n_cam * 3, int(param_len)),
        )
        if sparse.issparse(result):
            result = sparse.vstack((result, center_rows), format="csr")
        else:
            result = np.vstack((np.asarray(result), center_rows.toarray()))

    return result
