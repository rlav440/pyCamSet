"""
Whether a run is worth carrying into the next phase.

A phase can finish and still have produced nothing to go on: a camera that
detected the target in no image cannot be calibrated, and a solve that ends
at NaN has not solved anything.  Both were visible only as prose in a wall of
output, so the next phase was one green button away from being handed them.

Every phase's report already decides which of its own concerns are of that
kind -- see ``blocking_flags`` on the report classes -- so there is one rule
here and no second opinion about thresholds.
"""
from __future__ import annotations


def blocking_reasons(metadata: dict | None) -> list[str]:
    """
    Why the run in *metadata* should not be carried forward, if it should not.

    :param metadata: a phase's run record, as the runners save it
    :return: one sentence per reason, empty when the run is usable
    """
    if not metadata:
        return []

    reasons = []
    error = metadata.get("error")
    if error:
        reasons.append(str(error))

    report = metadata.get("report") or {}
    reasons.extend(str(flag) for flag in report.get("blocking_flags", []))
    quality_gate = (metadata.get("diagnostics") or {}).get("quality_gate") or {}
    reasons.extend(
        str(flag) for flag in quality_gate.get("blocking_flags", [])
        if str(flag) not in reasons
    )
    return reasons
