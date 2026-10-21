"""Guard the Phase 4 -> Phase 1 source-check recursion (R2).

``RunContext.phase3_state_source``.  Runs the real X/Y integration fixture
reused from ``tests/test_education_phase4.py``; the code under test is never
mocked.

handshake, not a sleep race; widen or shrink only with an equal witness.
"""
from __future__ import annotations

import copy
import importlib.util
import os
import pickle
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_xy_fixture_module():
    """Load the X/Y fixture helper by path.

    ``tests/`` is not an importable package, and importing it by module name
    would make pytest collect the file twice; loading it by path keeps this
    module's only dependency on it the helper itself.
    """
    path = _REPO_ROOT / "tests" / "test_education_phase4.py"
    spec = importlib.util.spec_from_file_location("_f2_xy_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_xy = _load_xy_fixture_module()

from pyCamSet.workflow import education_manifest as _manifest  # noqa: E402
from pyCamSet.workflow.education_trace import RunContext, Verification  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.data]

#: The in-progress disposition the nested call hands back when the cycle is cut.
SENTINEL_REASON = "Phase 1 manifest check is already in progress"

#: Upper bound on the handshake, so a missing sibling fails loudly instead of
#: hanging the run.  It is a deadline, never a race window.
HOLD_SECONDS = 5.0

#: Generous per-thread return deadline; a wrong verdict must fail, not hang.
JOIN_SECONDS = 120.0


# ---------------------------------------------------------------------------
# Shared real fixture: one X/Y workspace for the whole module.
#
# The helper is the integration fixture the card already ships, so the tests
# below reason about the production record shape rather than a stub.  It is
# module scoped only because building it is the expensive part (~20 s); the
# per-user configuration is redirected for the build so no test can edit the
# machine's own recently-used list.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def xy_workspace(tmp_path_factory):
    """Build the real X-recorded, Y-started Phase 4 workspace once."""
    base = tmp_path_factory.mktemp("f2-source-check")
    config = base / "user_config"
    config.mkdir()
    previous = os.environ.get("PYCAMSET_CONFIG_DIR")
    os.environ["PYCAMSET_CONFIG_DIR"] = str(config)
    try:
        (workspace, x_run_id, y_run_id, y_camset, _recorded,
         _records, _phase4_camset) = _xy._bound_xy_records(base)
    finally:
        if previous is None:
            os.environ.pop("PYCAMSET_CONFIG_DIR", None)
        else:
            os.environ["PYCAMSET_CONFIG_DIR"] = previous
    return {
        "workspace": workspace,
        "record": workspace.find_run("phase4", "p4-x-recorded-y-started"),
        "x_run_id": x_run_id,
        "y_run_id": y_run_id,
        "y_camset": y_camset,
    }


@pytest.fixture
def context(xy_workspace):
    """A fresh context per test: the guard state is per instance by design."""
    return RunContext.from_run(xy_workspace["workspace"], xy_workspace["record"])


@pytest.fixture
def reference(xy_workspace):
    """The single-threaded verdict every other result must match."""
    state = RunContext.from_run(
        xy_workspace["workspace"], xy_workspace["record"]).phase3_state_source()
    assert state.confirmed, "the X/Y fixture must resolve the used camset"
    assert state.verification is Verification.VERIFIED
    assert state.source_check == "verified by manifest"
    assert state.path == xy_workspace["y_camset"]
    assert state.origin_run_id == xy_workspace["y_run_id"]
    return state


# ---------------------------------------------------------------------------
# Probes.  Each is read-only instrumentation around the code under test.
# ---------------------------------------------------------------------------


@contextmanager
def sentinel_recorder():
    """Count every in-progress sentinel the cut constructs."""
    seen: list[str] = []
    real = _manifest.ManifestResult

    class Recording(real):  # type: ignore[misc, valid-type]
        def __init__(self, status, photo_names, unavailable, reason=None):
            super().__init__(status, photo_names, unavailable, reason)
            if reason == SENTINEL_REASON:
                seen.append(status)

    _manifest.ManifestResult = Recording
    try:
        yield seen
    finally:
        _manifest.ManifestResult = real


@contextmanager
def depth_probe():
    """Measure the deepest nesting of ``phase3_state_source``."""
    depth = {"now": 0, "max": 0}
    real = RunContext.phase3_state_source

    def counted(self):
        depth["now"] += 1
        depth["max"] = max(depth["max"], depth["now"])
        try:
            return real(self)
        finally:
            depth["now"] -= 1

    RunContext.phase3_state_source = counted
    try:
        yield depth
    finally:
        RunContext.phase3_state_source = real


def _marker_active(instance) -> bool:
    """Whether the re-entrancy marker on *instance* is currently set.

    Reads the marker however it is spelled, so the assertion holds for the
    per-thread holder, a copied mapping form, a lock, and a plain per-instance
    flag alike.
    """
    marker = getattr(instance, "_state_source_in_progress", None)
    if marker is None:
        return False
    if isinstance(marker, threading.local):
        return bool(getattr(marker, "active", False))
    if hasattr(marker, "locked"):
        return bool(marker.locked())
    if isinstance(marker, dict):
        return bool(marker.get("active", False))
    return bool(marker)


def guard_is_clear(instance) -> bool:
    """Whether no re-entrancy marker is left standing on *instance*."""
    return not _marker_active(instance)


@contextmanager
def phase1_handshake(hold_dir: Path):
    """Hold the guarded call until both threads are provably inside it.

    Each thread that reaches ``phase1_source_check`` publishes a marker file;
    the first to arrive waits (bounded) for the second.  A missing second
    arrival therefore becomes an assertion failure with a clear message
    instead of a silent race, and two arrivals mean the window really was
    concurrent rather than a lucky interleaving.
    """
    real = RunContext.phase1_source_check
    arrived: list[int] = []
    arrived_lock = threading.Lock()

    def held(self):
        ident = threading.get_ident()
        with arrived_lock:
            arrived.append(ident)
        marker = hold_dir / f"entered-{ident}.txt"
        marker.write_text("1", encoding="utf-8")
        deadline = time.monotonic() + HOLD_SECONDS
        while len(list(hold_dir.glob("entered-*.txt"))) < 2:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
        return real(self)

    RunContext.phase1_source_check = held
    try:
        yield arrived
    finally:
        RunContext.phase1_source_check = real


def run_concurrently(instance, workers=2):
    """Call ``phase3_state_source`` from *workers* threads at one barrier."""
    out: dict[int, object] = {}
    errors: dict[int, BaseException] = {}
    gate = threading.Barrier(workers)

    def call(index: int) -> None:
        try:
            gate.wait(timeout=30.0)
            out[index] = instance.phase3_state_source()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the test
            errors[index] = exc

    threads = [
        threading.Thread(target=call, args=(index,), name=f"f2-source-{index}")
        for index in range(workers)
    ]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=JOIN_SECONDS)
    assert not any(thread.is_alive() for thread in threads), (
        "a concurrent phase3_state_source call never returned")
    assert not errors, f"concurrent calls raised: {errors!r}"
    return out, time.monotonic() - started


# ---------------------------------------------------------------------------
# The regressions.
# ---------------------------------------------------------------------------


def test_phase3_state_source_cuts_the_phase1_recursion(reference, context):
    """The cycle is cut at the nested call and the outer verdict is recomputed."""
    with sentinel_recorder() as sentinels, depth_probe() as depth:
        state = context.phase3_state_source()

    assert depth["max"] == 2, (
        f"phase3_state_source nested {depth['max']} deep; the Phase 3 -> Phase 1 "
        "cycle must be cut at the first nested call")
    assert sentinels == ["could not be checked"], (
        "exactly one nested in-progress sentinel is expected per outer call")
    assert state.confirmed
    assert state.source_check == reference.source_check
    assert state.path == reference.path
    assert state.origin_run_id == reference.origin_run_id
    assert guard_is_clear(context)


def test_the_in_progress_sentinel_never_becomes_the_outer_source_check(
        reference, context):
    """The sentinel may only feed an intermediate value, never the verdict."""
    with sentinel_recorder() as sentinels:
        state = context.phase3_state_source()

    assert state.source_check == reference.source_check
    assert SENTINEL_REASON not in state.source_check
    assert len(sentinels) == 1, (
        f"expected the sentinel to be constructed once, saw {sentinels!r}")
    assert guard_is_clear(context)


def test_concurrent_calls_on_one_shared_context_agree_with_the_reference(
        reference, context, tmp_path):
    """Two threads on one shared RunContext must both get the real verdict.

    The handshake holds the guarded call open until both threads are inside
    it, so a plain per-instance flag is caught deterministically: the sibling
    takes the sentinel for its own outer verdict, and never reaches the
    guarded call at all.
    """
    hold_dir = tmp_path / "handshake"
    hold_dir.mkdir()
    with phase1_handshake(hold_dir) as arrived:
        with sentinel_recorder() as sentinels:
            out, elapsed = run_concurrently(context)

    verdicts = {state.source_check for state in out.values()}
    assert len(set(arrived)) == 2, (
        "both threads must reach the guarded Phase 1 call; only "
        f"{len(set(arrived))} did, so the guard suppressed a sibling call")
    assert verdicts == {reference.source_check}, (
        "a concurrent call on a shared RunContext disagreed with the "
        f"single-threaded verdict after {elapsed:.2f}s: {verdicts!r}")
    # Each thread cuts its own cycle once, so one sentinel per thread is the
    # expected count; the defect was a *shared* flag, not the sentinel itself.
    assert len(sentinels) == 2, (
        f"expected one nested sentinel per thread, saw {sentinels!r}")
    assert all(state.confirmed for state in out.values())
    assert guard_is_clear(context)


def test_concurrent_calls_leave_no_guard_residue_on_the_shared_context(
        reference, context, tmp_path):
    """The marker is released, and the same instance stays usable afterwards."""
    hold_dir = tmp_path / "handshake"
    hold_dir.mkdir()
    with phase1_handshake(hold_dir):
        run_concurrently(context)

    assert guard_is_clear(context), "the re-entrancy marker was left standing"
    again = context.phase3_state_source()
    assert again.confirmed
    assert again.source_check == reference.source_check
    assert guard_is_clear(context)


def test_repeated_calls_on_one_shared_context_are_deterministic(
        reference, context):
    """Sequential repeats on one instance are byte-identical."""
    keys = set()
    for _ in range(3):
        state = context.phase3_state_source()
        keys.add((state.confirmed, state.source_check, str(state.path),
                  state.origin_run_id, state.phase3_fixed_pose_ordinal1,
                  state.phase4_fixed_pose_ordinal1, state.start_metric_px))

    assert keys == {(reference.confirmed, reference.source_check,
                     str(reference.path), reference.origin_run_id,
                     reference.phase3_fixed_pose_ordinal1,
                     reference.phase4_fixed_pose_ordinal1,
                     reference.start_metric_px)}
    assert guard_is_clear(context)


def test_an_interrupted_manifest_check_resets_the_guard(monkeypatch, context):
    """A BaseException from the guarded call propagates and releases the marker."""
    real = RunContext.phase1_source_check

    def interrupted(self):
        raise KeyboardInterrupt("simulated interruption of the manifest check")

    monkeypatch.setattr(RunContext, "phase1_source_check", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            context.phase3_state_source()
        assert guard_is_clear(context), (
            "the marker must be released in a finally, not only on Exception")
    finally:
        monkeypatch.setattr(RunContext, "phase1_source_check", real)

    state = context.phase3_state_source()
    assert state.confirmed
    assert guard_is_clear(context)


# ---------------------------------------------------------------------------
# RC1: copy/pickle lifecycle.  The re-entrancy cut must survive serialization
# without regressing the per-instance / per-thread isolation.
# ---------------------------------------------------------------------------


def _assert_marker_idle(instance, where: str, *, source=None) -> None:
    """A deserialized context carries a fresh, idle, unshared guard marker."""
    marker = getattr(instance, "_state_source_in_progress", None)
    assert marker is not None, f"{where}: the guard marker must exist after round-trip"
    assert not _marker_active(instance), (
        f"{where}: the fresh guard marker must start idle, not inherit a cut")
    if source is not None:
        sibling = getattr(source, "_state_source_in_progress", None)
        assert marker is not sibling, (
            f"{where}: the deserialized context must not share its guard marker "
            "with the original instance")


def test_deepcopy_round_trip_of_an_idle_context_keeps_the_guard_idle(context):
    """``copy.deepcopy`` must not fail on the guard (RC1) and must stay idle."""
    _ = context  # build the real context; the marker lives on the instance
    clone = copy.deepcopy(context)

    assert clone is not context
    assert clone.run_id == context.run_id
    assert clone.phase == context.phase
    _assert_marker_idle(clone, "deepcopy", source=context)

    # the clone's cut must be a real, independent per-instance guard: a call on
    # the clone resolves the same verdict and leaves no residue on either.
    state = clone.phase3_state_source()
    assert state.confirmed
    assert guard_is_clear(clone)
    assert guard_is_clear(context)


def test_pickle_round_trip_of_an_idle_context_keeps_the_guard_idle(context):
    """``pickle`` must not fail on the guard (RC1) and must stay idle."""
    payload = pickle.dumps(context)
    clone = pickle.loads(payload)

    assert clone.run_id == context.run_id
    assert clone.phase == context.phase
    _assert_marker_idle(clone, "pickle", source=context)

    state = clone.phase3_state_source()
    assert state.confirmed
    assert guard_is_clear(clone)
    assert guard_is_clear(context)


def test_deepcopy_during_an_active_cut_does_not_freeze_the_clone(context):
    """A snapshot taken while a cut is active must not carry the cut over.

    The cut is process-local and thread-local; a serialized snapshot therefore
    has to arrive with a fresh, idle guard.  If the holder were the per-thread
    ``threading.local`` this ``deepcopy`` would raise ``TypeError`` instead.
    """
    real = RunContext.phase1_source_check
    observed: dict[str, object] = {}

    def snapshotting(self):
        observed["clone"] = copy.deepcopy(self)
        return real(self)

    RunContext.phase1_source_check = snapshotting
    try:
        state = context.phase3_state_source()
    finally:
        RunContext.phase1_source_check = real

    assert state.confirmed
    clone = observed["clone"]
    assert clone.run_id == context.run_id
    _assert_marker_idle(clone, "deepcopy-during-cut", source=context)
    assert guard_is_clear(context)


def test_pickle_during_an_active_cut_does_not_freeze_the_clone(context):
    """The pickle form of the same snapshot invariant as the deepcopy case."""
    real = RunContext.phase1_source_check
    observed: dict[str, object] = {}

    def snapshotting(self):
        observed["blob"] = pickle.dumps(self)
        return real(self)

    RunContext.phase1_source_check = snapshotting
    try:
        state = context.phase3_state_source()
    finally:
        RunContext.phase1_source_check = real

    assert state.confirmed
    clone = pickle.loads(observed["blob"])
    assert clone.run_id == context.run_id
    _assert_marker_idle(clone, "pickle-during-cut", source=context)
    assert guard_is_clear(context)


def test_serialization_round_trip_preserves_per_thread_and_per_instance_isolation(
        reference, context):
    """RC1 must not weaken RC1's own target: the per-thread/per-instance cut.

    Both requirements hold together on a deserialized context: two threads on
    one shared clone reach the guarded call and agree with the reference, and
    the clone's marker is distinct from the original's.
    """
    clone = pickle.loads(pickle.dumps(context))
    assert clone.run_id == context.run_id

    from_origin = context._state_source_in_progress
    from_clone = clone._state_source_in_progress
    assert (from_origin is not from_clone if isinstance(from_origin, threading.local)
            else id(from_origin) != id(from_clone)), (
        "a deserialized context must not share its guard marker with the original")

    hold_dir = _serialization_hold_dir()
    with phase1_handshake(hold_dir) as arrived:
        out, elapsed = run_concurrently(clone)

    assert len(set(arrived)) == 2, (
        "both threads on the deserialized context must reach the guarded call")
    assert {state.source_check for state in out.values()} == {reference.source_check}, (
        f"a concurrent call on a deserialized context disagreed after {elapsed:.2f}s")
    assert all(state.confirmed for state in out.values())
    assert guard_is_clear(clone)
    assert guard_is_clear(context)


def _serialization_hold_dir() -> Path:
    import tempfile
    return Path(tempfile.mkdtemp(prefix="f2-serialization-"))

