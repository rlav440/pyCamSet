'''Collect, validate and project bounded solver-step traces for education playback.
'''
from __future__ import annotations

import importlib.metadata
import bisect
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


MAX_INTERIOR_FRAMES = 100
MAX_TRACE_BUDGET = 8 * 1024 * 1024
COST_REPRODUCTION_TOLERANCE = 1e-8


def displayed_geometry(handler: Any, params: np.ndarray) -> np.ndarray:
    """Return camera/pose/point translations in the solver's display length units."""
    primitive = handler.bundlePrimitive
    # The primitive mutates arrays shared with the solver handler, so preserve
    # and restore them around this display-only projection.
    names = ("intr", "extr", "poses", "bundle_pts")
    originals = {name: getattr(primitive, name) for name in names
                 if hasattr(primitive, name)}
    snapshots = {name: value.copy() for name, value in originals.items()}
    try:
        values = primitive.return_bundle_primitives(
            np.asarray(params, dtype=np.float64).copy())
        # Displayed motion is translational: exclude rotation and intrinsics,
        # which use different units and would skew frame selection.
        geometry = [np.asarray(values[1], dtype=np.float64)[:, 3:6].reshape(-1),
                    np.asarray(values[2], dtype=np.float64)[:, 3:6].reshape(-1)]
        if len(values) > 3:
            geometry.append(np.asarray(values[3], dtype=np.float64).reshape(-1))
        return np.concatenate(geometry).copy()
    finally:
        for name, value in originals.items():
            value[...] = snapshots[name]


def _motion(left: np.ndarray, right: np.ndarray) -> float:
    """Measure translation/point displacement in the displayed geometry units."""
    return float(np.linalg.norm(right - left))


def _thin_by_motion(frames: list[tuple], max_count: int) -> list[tuple]:
    """Choose a bounded subsequence minimising the largest displayed-motion gap."""
    if len(frames) <= max_count:
        return frames
    coordinates = [float(frame[3]) for frame in frames]
    # Binary-search the smallest feasible maximum arc-length gap. This uses
    # only already bounded candidate frames and always preserves both ends.
    gaps = sorted({coordinates[j] - coordinates[i]
                   for i in range(len(coordinates))
                   for j in range(i + 1, len(coordinates))})

    def select(gap_limit: float) -> list[int] | None:
        chosen = [0]
        while chosen[-1] < len(coordinates) - 1:
            next_index = bisect.bisect_right(
                coordinates, coordinates[chosen[-1]] + gap_limit) - 1
            if next_index <= chosen[-1]:
                return None
            chosen.append(next_index)
            if len(chosen) > max_count:
                return None
        return chosen

    low, high = 0, len(gaps) - 1
    best: list[int] | None = None
    while low <= high:
        middle = (low + high) // 2
        candidate = select(gaps[middle])
        if candidate is None:
            low = middle + 1
        else:
            best = candidate
            high = middle - 1
    if best is None:
        return [frames[0], frames[-1]]
    return [frames[index] for index in best]


@dataclass
class AcceptedStepCollector:
    """Keep endpoints and a bounded motion-aware subset of accepted iterates."""
    handler: Any
    initial: np.ndarray
    max_frames: int = MAX_INTERIOR_FRAMES
    byte_budget: int = MAX_TRACE_BUDGET
    initial_residuals: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.initial = np.asarray(self.initial, dtype=np.float64).copy()
        self.endpoints_only = False
        self._frames: list[tuple[int, float, np.ndarray]] = []
        self._callback_count = 0
        self._start_geometry = displayed_geometry(self.handler, self.initial)
        self._initial_cost = (
            _cost_from_residuals(self.handler, self.initial_residuals)
            if self.initial_residuals is not None else None)
        self._last_x = self.initial.copy()
        self._last_motion = self._start_geometry.copy()
        budget_candidates = self.byte_budget // max(1, self.initial.nbytes * 3)
        self._candidate_limit = max(2, min(self.max_frames * 2 + 2,
                                            budget_candidates))
        self._cumulative_motion = 0.0

    def add_schur(self, nit: int, cost: float, params: np.ndarray) -> None:
        self._add(nit, cost, params)

    def scipy_callback(self, intermediate_result: Any) -> None:
        """Accept both SciPy callback conventions and drop unchanged rejected trials."""
        if hasattr(intermediate_result, "x"):
            params = np.asarray(intermediate_result.x, dtype=np.float64).copy()
            nit = int(getattr(intermediate_result, "nit", len(self._frames) + 1))
            cost = (float(intermediate_result.cost)
                    if hasattr(intermediate_result, "cost")
                    else _cost_at_start(self.handler, params))
        else:
            params = np.asarray(intermediate_result, dtype=np.float64).copy()
            nit = self._callback_count + 1
            cost = float(_cost_at_start(self.handler, params))
        if np.array_equal(params, self._last_x):
            return
        self._callback_count += 1
        self._add(nit, cost, params)

    def _add(self, nit: int, cost: float, params: np.ndarray) -> None:
        snapshot = np.asarray(params, dtype=np.float64).copy()
        if np.array_equal(snapshot, self._last_x):
            return
        geometry = displayed_geometry(self.handler, snapshot)
        self._cumulative_motion += _motion(self._last_motion, geometry)
        self._frames.append((int(nit), float(cost), snapshot,
                             self._cumulative_motion))
        self._last_x = snapshot.copy()
        self._last_motion = geometry
        if len(self._frames) > self._candidate_limit * 2:
            self._thin_candidates()

    def _thin_candidates(self) -> None:
        """Retain motion landmarks, avoiding cost-drop-biased geometry thinning."""
        if len(self._frames) <= self._candidate_limit * 2:
            return
        start = (0, _cost_at_start(self.handler, self.initial),
                 self.initial.copy(), 0.0)
        self._frames = _thin_by_motion(
            [start, *self._frames], self._candidate_limit + 1)[1:]

    def finish(self, params: np.ndarray, cost: float, nit: int = 0) -> None:
        """Set the exact final solver vector as the protected endpoint."""
        final = np.asarray(params, dtype=np.float64).copy()
        if self._frames and np.array_equal(final, self._frames[-1][2]):
            return
        self._add(int(nit), float(cost), final)

    def to_payload(self, *, solver: str, problem_opts: dict) -> dict:
        """Serialize float64 endpoints, bounded interior states and provenance."""
        initial_cost = (self._initial_cost if self._initial_cost is not None
                        else _cost_at_start(self.handler, self.initial))
        frames = [(0, initial_cost,
                   self.initial.copy(), 0.0)]
        frames.extend(self._frames)
        # Reject costs that cannot be represented by the canonical JSON writer.
        if any(not math.isfinite(frame[1]) for frame in frames):
            raise ValueError("solver step trace contains a non-finite cost")
        byte_cost = sum(frame[2].nbytes * 3 for frame in frames)
        weight = "displayed_translation_motion_uniform_arc_length"
        if byte_cost > self.byte_budget:
            # Keep endpoints in float64 and label any interior precision fallback.
            interior = frames[1:-1]
            frames = [frames[0], *[(n, c, x.astype(np.float32), arc)
                                   for n, c, x, arc in interior], frames[-1]]
            precision = "float32_interiors_due_to_budget"
        else:
            precision = "float64"
        # Reduce interior count further when JSON's conservative 3x estimate binds.
        max_kept = max(2, min(self.max_frames + 2,
                              self.byte_budget // max(1, self.initial.nbytes * 3)))
        frames = _thin_by_motion(frames, max_kept)
        return {
            "steps": [{"nit": int(n), "cost": float(c), "x": x.tolist(),
                       "dtype": str(x.dtype)} for n, c, x, _arc in frames],
            "loss": str(problem_opts.get("loss", "linear")),
            "f_scale": float(problem_opts.get("f_scale", 1.0)),
            "solver": solver,
            "thinning_weight": weight,
            "interior_frame_cap": int(self.max_frames),
            "byte_budget": int(self.byte_budget),
            "byte_budget_counts_endpoints": True,
            "endpoint_bytes": int(frames[0][2].nbytes + frames[-1][2].nbytes),
            "budget_exceeded_by_endpoints": bool(
                frames[0][2].nbytes + frames[-1][2].nbytes > self.byte_budget),
            "budget_precision": precision,
            "endpoints_only": bool(self.endpoints_only),
            "scipy_version": _version("scipy"),
            "numpy_version": np.__version__,
            "opencv_major_version": int(cv2.__version__.split(".")[0]),
            "motion_units": "target coordinate units; translation only",
            "cost_reproduction_tolerance": COST_REPRODUCTION_TOLERANCE,
        }


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _cost_at_start(handler: Any, params: np.ndarray) -> float:
    residuals = np.asarray(handler.make_loss_fun(1)(params), dtype=np.float64)
    return _cost_from_residuals(handler, residuals)


def _cost_from_residuals(handler: Any, residuals: np.ndarray) -> float:
    """Compute the trace objective from residuals already evaluated by the solve."""
    residuals = np.asarray(residuals, dtype=np.float64)
    loss = str(getattr(handler, "problem_opts", {}).get("loss", "linear"))
    if loss != "linear":
        from pyCamSet.optimisation.robust_loss import _transform
        residuals, _ = _transform(
            residuals, loss,
            float(getattr(handler, "problem_opts", {}).get("f_scale", 1.0)))
    return 0.5 * float(residuals @ residuals)


def read_accepted_step_trace(trace: dict, *, expected_start: np.ndarray,
                             saved_params: np.ndarray,
                             handler: Any | None = None) -> dict:
    """Validate all frames and exact endpoints before allowing playback."""
    if trace.get("status") != "available":
        return {"status": "unavailable", "reason": trace.get("reason", "trace unavailable")}
    records = [record for record in trace.get("records", [])
               if record.get("kind") == "accepted_step"]
    if not records:
        return {"status": "unavailable", "reason": "no accepted steps"}
    payload = records[-1].get("payload", {})
    if payload.get("endpoints_only"):
        return {"status": "unavailable", "reason": "steps not recorded for this solver"}
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return {"status": "unavailable", "reason": "no accepted steps"}
    try:
        arrays = [np.asarray(row["x"], dtype=row.get("dtype", "float64")) for row in steps]
        costs = [float(row["cost"]) for row in steps]
        nits = [int(row["nit"]) for row in steps]
        expected = np.asarray(expected_start, dtype=np.float64)
        final = np.asarray(saved_params, dtype=np.float64)
        if not np.array_equal(arrays[0], expected) or not np.array_equal(arrays[-1], final):
            raise ValueError("recorded endpoints do not match saved parameters")
        if any(right <= left for left, right in zip(nits, nits[1:])):
            raise ValueError("accepted-step nit values are not strictly increasing")
        if any(right > left for left, right in zip(costs, costs[1:])):
            raise ValueError("accepted-step costs are not non-increasing")
        if handler is not None:
            loss_fn = handler.make_loss_fun(1)
            loss_name = str(payload.get("loss", "linear"))
            f_scale = float(payload.get("f_scale", 1.0))
            tolerance = float(payload.get(
                "cost_reproduction_tolerance", COST_REPRODUCTION_TOLERANCE))
            for params, stored_cost in zip(arrays, costs):
                residuals = np.asarray(loss_fn(np.asarray(params, dtype=np.float64)), dtype=np.float64)
                if loss_name != "linear":
                    from pyCamSet.optimisation.robust_loss import _transform
                    residuals, _ = _transform(residuals, loss_name, f_scale)
                actual_cost = 0.5 * float(residuals @ residuals)
                if not np.isclose(actual_cost, stored_cost, rtol=tolerance, atol=tolerance):
                    raise ValueError("stored accepted-step cost does not reproduce from x")
        return {"status": "available", "payload": payload, "arrays": arrays,
                "nits": nits, "costs": costs,
                "costs_checked": handler is not None}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def prepare_saved_phase3_handler(handler: Any, params: np.ndarray) -> None:
    """Restore outlier-excluded pose slots on a private, reloaded handler.

    Older saved handlers can carry a fixed-pose-only primitive mask even
    though their solver vector and missing-pose mask exclude more images.
    This changes only the fresh in-memory replay handler, never a saved run.
    """
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler
    if not isinstance(handler, TemplateBundleHandler):
        return  # Pure synthetic frame-builder tests use minimal stand-ins.
    missing = np.asarray(handler.missing_poses, dtype=bool).reshape(-1)
    fixed = int(handler.problem_opts.get("fixed_pose", 0))
    if not len(missing) or fixed < 0 or fixed >= len(missing) or missing[fixed]:
        raise ValueError("saved Phase 3 reference pose is unusable")
    primitive = handler.bundlePrimitive
    mask = ~missing.copy()
    mask[fixed] = False
    if np.asarray(primitive.poses_unfixed).shape != mask.shape:
        raise ValueError("saved Phase 3 pose mask layout is inconsistent")
    primitive.poses_unfixed = mask
    primitive.calc_free_poses()
    if primitive.pose_end != len(params):
        raise ValueError("saved Phase 3 pose mask does not match its vector")


def build_phase3_frames(handler_factory: Any, trace: dict, *, expected_start: np.ndarray,
                        saved_params: np.ndarray) -> dict:
    """Build frames from a factory-created fresh handler, never an in-use CameraSet."""
    handler = handler_factory()
    try:
        prepare_saved_phase3_handler(handler, np.asarray(expected_start))
    except (ValueError, TypeError, AttributeError):
        return {"status": "unavailable", "reason": "saved Phase 3 pose layout does not match recorded steps"}
    validated = read_accepted_step_trace(
        trace, expected_start=expected_start, saved_params=saved_params,
        handler=handler)
    if validated["status"] != "available":
        return validated
    frames = []
    for params in validated["arrays"]:
        primitives = handler.bundlePrimitive.return_bundle_primitives(
            np.asarray(params, dtype=np.float64).copy())
        frames.append(tuple(np.asarray(value).copy() for value in primitives))
    return {**validated, "frames": frames}
