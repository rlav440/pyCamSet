"""Purpose: Optional Optuna integration for the Optimisation tab.

Status: Active adapter that keeps Optuna optional at import time.

Future: Keep the public surface small so the headless worker remains testable.

``optuna`` is an *optional* dependency.  Importing this module never raises;
callers should consult :data:`OPTUNA_AVAILABLE` and, if it is ``False``, fall
back to a non-Optuna sampler or surface a helpful message to the user.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from pyCamSet.optimisation.charuco_detector_metadata import (
    coerce_value,
    metadata_by_key,
)
from pyCamSet.optimisation.optimisation_worker import ParameterRowConfig

_LOG = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by environments with/without optuna
    import optuna  # type: ignore[import-not-found]

    OPTUNA_AVAILABLE = True
    OPTUNA_VERSION = getattr(optuna, "__version__", "unknown")
except Exception:  # pragma: no cover - defensive; importing optuna should not fail loudly
    optuna = None  # type: ignore[assignment]
    OPTUNA_AVAILABLE = False
    OPTUNA_VERSION = None


def require_optuna() -> None:
    """Raise :class:`ImportError` with a helpful message when optuna is missing."""
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "optuna is not installed. Install it with `pip install optuna` to use "
            "the Optimisation tab's sampling features."
        )


def make_optuna_sampler(
    *,
    seed: Optional[int] = None,
    sampler_name: Optional[str] = None,
) -> Any:
    """Build an Optuna sampler.  ``sampler_name`` is a class name on ``optuna.samplers``."""
    require_optuna()
    if not sampler_name:
        return optuna.samplers.TPESampler(seed=seed)
    cls = getattr(optuna.samplers, sampler_name, None)
    if cls is None:
        _LOG.warning("Unknown sampler %r; falling back to TPESampler", sampler_name)
        return optuna.samplers.TPESampler(seed=seed)
    try:
        return cls(seed=seed)
    except TypeError:
        return cls()


def suggest_for_row(trial: Any, row: ParameterRowConfig) -> Optional[Any]:
    """Sample one parameter from an Optuna trial, honouring its dtype.

    Returns ``None`` when *row* is not optimised; otherwise an int or float.
    """
    if not row.optimise:
        return None
    entry = metadata_by_key().get(row.key)
    if entry is None:
        return None
    choices = [choice["value"] for choice in entry.get("choices", [])]
    if choices:
        return trial.suggest_categorical(row.key, choices)
    if row.lower is None or row.upper is None:
        return None
    lo = coerce_value(entry, row.lower)
    hi = coerce_value(entry, row.upper)
    if lo > hi:
        lo, hi = hi, lo
    if entry["dtype"] == "int":
        return int(trial.suggest_int(row.key, int(lo), int(hi)))
    return float(trial.suggest_float(row.key, float(lo), float(hi)))


def build_optuna_sampler_callable(
    *,
    study: Any,
) -> Callable[[int, list[ParameterRowConfig]], dict[str, Any]]:
    """Return a sampler callable suitable for ``OptimisationStudy(sampler=...)``.

    The callable spawns a fresh Optuna trial inside *study* and returns one
    sampled override dict per worker trial.  Values may be ints, floats, or
    discrete categorical codes such as ``cornerRefinementMethod``.

    Note: most users should prefer :func:`run_optuna_study`, which owns the
    full ask/tell cycle.
    """
    require_optuna()
    state: dict[str, Any] = {"trial": None, "params": {}}

    def _sampler(trial_number: int, rows: list[ParameterRowConfig]) -> dict[str, Any]:
        trial = study.ask()
        state["trial"] = trial
        sampled: dict[str, Any] = {}
        for row in rows:
            value = suggest_for_row(trial, row)
            if value is not None:
                sampled[row.key] = value
        state["params"] = sampled
        return sampled

    _sampler.state = state  # type: ignore[attr-defined]
    return _sampler


def run_optuna_study(
    *,
    config: Any,  # RunConfig — typed loosely to avoid a circular import
    detection_fn,
    phase2_fn=None,
    phase3_fn=None,
    phase4_fn=None,
    cancel_token=None,
    progress_cb=None,
    write_metadata: bool = True,
) -> Any:
    """Run an Optimisation study driven by Optuna.

    Returns the :class:`pyCamSet.optimisation.optimisation_worker.SuccessRetention`
    populated by the run.
    """
    from pyCamSet.optimisation.optimisation_worker import (
        OptimisationStudy,
        run_trial,
        fixed_settings_only,
        detection_options_from_settings,
    )
    require_optuna()

    sampler = make_optuna_sampler(seed=config.seed, sampler_name=config.sampler_name)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Wrap the worker so each ask/tell round runs through one trial.
    driver = OptimisationStudy(
        config,
        sampler=lambda i, rows: {},  # placeholder; real sampling inside the loop
        detection_fn=detection_fn,
        phase2_fn=phase2_fn,
        phase3_fn=phase3_fn,
        phase4_fn=phase4_fn,
        cancel_token=cancel_token,
        progress_cb=progress_cb,
        write_metadata=write_metadata,
    )
    errors = driver.validate()
    if errors:
        raise ValueError("Run configuration invalid: " + "; ".join(errors))

    if config.mode == "full" and driver.baseline_point_count is None:
        driver.compute_baseline()

    output_dir = config.resolved_output_dir(driver.study_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = driver._metadata_writer_for(output_dir)  # noqa: SLF001 (internal helper)
    import time
    started = time.time()
    driver._started_at = started  # noqa: SLF001
    completed = 0

    for trial_number in range(config.n_trials):
        if driver.cancel_token.is_cancelled():
            break
        trial = study.ask()
        sampled: dict[str, Any] = {}
        for row in config.parameter_rows:
            value = suggest_for_row(trial, row)
            if value is not None:
                sampled[row.key] = value
        result, _payload = run_trial(
            trial_number,
            config.parameter_rows,
            config=config,
            sampled=sampled,
            baseline_point_count=driver.baseline_point_count,
            detection_fn=detection_fn,
            phase2_fn=phase2_fn,
            phase3_fn=phase3_fn,
            phase4_fn=phase4_fn,
            metadata_writer=writer,
        )
        driver.results.append(result)
        driver.retention.consider(result)
        completed += 1
        study.tell(trial, result.score)
        if progress_cb is not None:
            from pyCamSet.optimisation.optimisation_worker import StudyProgress
            best = driver.retention.best()
            progress_cb(
                StudyProgress(
                    trial_number=trial_number,
                    total_trials=config.n_trials,
                    best_score=best.score if best else float("inf"),
                    best_stage=best.success_stage if best else None,
                    best_phase3_rpe=best.phase3_rpe if best else None,
                    best_phase4_rpe=best.phase4_rpe if best else None,
                    n_successes=len(driver.retention),
                    elapsed_sec=time.time() - started,
                    last_result=result,
                )
            )

    driver._finished_at = time.time()  # noqa: SLF001
    driver.retention.n_completed = completed  # type: ignore[attr-defined]
    if write_metadata:
        from pyCamSet.optimisation.optimisation_study import write_study_summary
        write_study_summary(
            output_dir,
            study_id=driver.study_id,
            started_at=started,
            finished_at=driver._finished_at,  # noqa: SLF001
            retention=driver.retention,
            n_trials_completed=completed,
            n_trials_requested=config.n_trials,
            mode=config.mode,
            sampler_name=config.sampler_name,
            seed=config.seed,
        )
    return driver.retention


__all__ = [
    "OPTUNA_AVAILABLE",
    "OPTUNA_VERSION",
    "require_optuna",
    "make_optuna_sampler",
    "suggest_for_row",
    "build_optuna_sampler_callable",
    "run_optuna_study",
]
