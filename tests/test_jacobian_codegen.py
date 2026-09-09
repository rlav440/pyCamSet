"""Guards on the generated analytic jacobian.

The bundle adjuster's jacobian is Python source generated at runtime.  Its
sparsity is inferred by evaluating each function block numerically and encoding
entries that equal 0 or 1 as literals.  Inferring that from a single sample
bakes a coincidental zero into the emitted source, which then drops those
derivative terms for every future input -- silently, with no error, producing a
worse calibration rather than a failure.

That happened: on macos-14 the rotation half of every target pose was encoded
as structurally zero (69 of 183 columns), and the Ccube calibration stalled at
6.17 px where x86_64 reached 2.62 px.  Feeding the solver a numeric jacobian on
the same machine recovered 2.62 px, which is what identified the generated
jacobian as the cause.

The fast tests here cover the encoding rule directly; the data-backed test
checks the real generated jacobian against finite differences.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from pyCamSet.optimisation.matmul_map import (
    SPARSITY_PROBE_SAMPLES,
    convert_matrix,
    indx,
)

# --------------------------------------------------------------------------
# Fast: the 0/1 encoding rule
# --------------------------------------------------------------------------


def test_entry_zero_in_only_one_sample_is_not_encoded_as_zero():
    """The regression. A derivative that vanishes at one probe point is not
    structurally zero, and must still be emitted."""
    samples = np.ones((3, 2, 2))
    samples[1, 0, 1] = 0.0  # vanishes in exactly one sample

    encoded = convert_matrix(samples, size=2, matnum=0)

    assert isinstance(encoded[(0, 1)], indx), (
        "an entry that is zero in only some samples was encoded as a "
        "structural zero, which drops the term from the generated jacobian"
    )


def test_entry_zero_in_every_sample_is_encoded_as_zero():
    """A genuinely structural zero should still be elided from the source."""
    samples = np.ones((3, 2, 2))
    samples[:, 0, 1] = 0.0

    encoded = convert_matrix(samples, size=2, matnum=0)

    assert encoded[(0, 1)] == 0


def test_entry_one_in_every_sample_is_encoded_as_one():
    """Ones are the multiplicative identity and are elided the same way."""
    samples = np.full((4, 2, 2), 7.0)
    samples[:, 1, 1] = 1.0

    encoded = convert_matrix(samples, size=2, matnum=0)

    assert encoded[(1, 1)] == 1
    assert isinstance(encoded[(0, 0)], indx)


def test_entry_one_in_only_one_sample_is_not_encoded_as_one():
    samples = np.full((3, 2, 2), 7.0)
    samples[2, 1, 0] = 1.0

    encoded = convert_matrix(samples, size=2, matnum=0)

    assert isinstance(encoded[(1, 0)], indx)


def test_nan_is_never_encoded_as_a_constant():
    """The probe fills its output buffer with NaN, so an entry a compiled
    kernel fails to write is emitted rather than silently dropped."""
    samples = np.full((2, 2, 2), np.nan)

    encoded = convert_matrix(samples, size=2, matnum=0)

    assert all(isinstance(v, indx) for v in encoded.values())


def test_single_unstacked_sample_is_tolerated():
    """convert_matrix historically took one matrix; keep that working."""
    encoded = convert_matrix(np.array([[0.0, 1.0], [2.0, 3.0]]), size=2, matnum=0)

    assert encoded[(0, 0)] == 0
    assert encoded[(0, 1)] == 1
    assert isinstance(encoded[(1, 1)], indx)


def test_probe_takes_enough_samples_to_be_meaningful():
    """One sample is what caused the macOS failure."""
    assert SPARSITY_PROBE_SAMPLES >= 10


# --------------------------------------------------------------------------
# Fast: the probe must feed each kernel a full input buffer
# --------------------------------------------------------------------------


def test_template_block_reads_three_inputs_despite_declaring_none():
    """``template_points`` declares ``num_inp = 0`` but reads ``inp[0:3]``.

    At runtime that is fine: the generated code allocates one shared input
    buffer and writes the template point into ``inp[:3]``.  But the codegen
    probe used to size its buffer from ``num_inp``, handing this kernel a
    zero-length array.  Numba does not bounds check, so the kernel
    differentiated whatever followed in memory -- nonzero heap garbage on
    x86_64, zeros on arm64.  The zeros were then encoded as structural zeros
    and the rotation derivatives vanished from the generated jacobian.

    This pins the behaviour the probe has to respect.
    """
    from pyCamSet.optimisation.function_block_implementations import template_points

    params = np.array([0.3, 0.4, 0.5, 1.0, 2.0, 3.0])

    def rotation_block(inp):
        output = np.full(18, np.nan)
        template_points.compute_jac(
            params=params, inp=inp, output=output, memory=np.zeros(27)
        )
        return output.reshape(3, 6)[:, :3]

    # A real template point gives real rotation derivatives.
    assert np.any(rotation_block(np.array([0.7, 0.2, 0.9])) != 0)

    # The failure mode: zeros in, and the whole rotation block reads as
    # structurally zero.
    assert np.all(rotation_block(np.zeros(3)) == 0)


def test_read_width_is_declared_not_inferred_from_num_inp():
    """``n_inp_read`` must cover what the kernel reads, even alone.

    Without it ``template_points`` reports a width of 0, which is what handed
    the probe a zero-length buffer.
    """
    from pyCamSet.optimisation.function_block_implementations import (
        extrinsic3D,
        free_point,
        projection,
        template_points,
    )
    from pyCamSet.optimisation.matmul_map import input_buffer_width

    assert input_buffer_width([template_points]) >= 3
    assert input_buffer_width([projection, extrinsic3D, template_points]) >= 3
    # free_point never touches inp, so it needs nothing -- which is why the
    # self-calibration composition was unaffected by this bug.
    assert input_buffer_width([free_point]) == 0


def test_bounds_check_rejects_an_undersized_input_buffer():
    """The generic guard: any kernel reading past its buffer must be caught.

    numba compiles without bounds checking, so this cross-checks against the
    pure-Python original, which numpy does bounds check.
    """
    from pyCamSet.optimisation.function_block_implementations import (
        free_point,
        template_points,
    )
    from pyCamSet.optimisation.matmul_map import check_block_stays_in_bounds

    for too_narrow in (0, 2):
        with pytest.raises(RuntimeError, match="reads outside"):
            check_block_stays_in_bounds(template_points, too_narrow)

    check_block_stays_in_bounds(template_points, 3)  # wide enough
    check_block_stays_in_bounds(free_point, 0)  # never reads inp


# --------------------------------------------------------------------------
# Data-backed: the real generated jacobian
# --------------------------------------------------------------------------


def _build_ccube_problem(data_dir: Path):
    """Detect, initialise, and return the loss/jacobian for a Ccube problem."""
    from cv2 import aruco

    from pyCamSet import Ccube
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile,
        run_initial_calibration,
    )
    from pyCamSet.optimisation.optimisation_handling import make_optimisation_function
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    loc = data_dir / "calibration_ccube"
    target = Ccube(
        n_points=10, length=40, aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2
    )
    detections, camera_res = detect_datapoints_in_imfile(
        f_loc=loc, caching=False, calibration_target=target, threads=1
    )
    cams = run_initial_calibration(detections, target, camera_res, save=False)
    cams.set_resolutions_from_file(floc=loc)

    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detections,
        options={"outliers": "n"},
    )
    return make_optimisation_function(handler, 1)


def _dense(jac):
    return np.asarray(jac.todense()) if hasattr(jac, "todense") else np.asarray(jac)


@pytest.fixture(scope="module")
def ccube_problem(request):
    """The built Ccube problem, shared across the data-backed tests."""
    data_dir = Path(__file__).resolve().parent / "test_data"
    if not data_dir.is_dir():
        pytest.skip(f"image corpus not found at {data_dir}")
    return _build_ccube_problem(data_dir)


@pytest.mark.data
@pytest.mark.slow
def test_generated_jacobian_has_no_all_zero_columns(ccube_problem):
    """Every parameter must influence the residuals.

    On macos-14 this was 69 zero columns -- 23 target poses x 3 rotation
    parameters -- so the solver could not rotate the target at all.
    """
    _loss_fn, jac_fn, init_params = ccube_problem
    jac = _dense(jac_fn(init_params))

    zero_columns = np.flatnonzero(np.linalg.norm(jac, axis=0) == 0.0)
    assert zero_columns.size == 0, (
        f"{zero_columns.size} of {jac.shape[1]} jacobian columns are "
        f"identically zero: {zero_columns.tolist()}. The generated jacobian "
        "has dropped derivative terms, so these parameters cannot be "
        "optimised."
    )


@pytest.mark.data
@pytest.mark.slow
def test_generated_jacobian_matches_finite_differences(ccube_problem):
    """The analytic jacobian must agree with the loss function it differentiates."""
    loss_fn, jac_fn, init_params = ccube_problem
    jac = _dense(jac_fn(init_params))
    base = np.asarray(loss_fn(init_params), dtype=float)

    rng = np.random.default_rng(0)  # same columns on every platform
    columns = rng.choice(init_params.size, size=16, replace=False)

    worst_col, worst_err = None, 0.0
    for col in columns:
        step = 1e-6 * max(1.0, abs(init_params[col]))
        shifted = init_params.copy()
        shifted[col] += step
        numeric = (np.asarray(loss_fn(shifted), dtype=float) - base) / step
        rel = np.linalg.norm(jac[:, col] - numeric) / max(np.linalg.norm(numeric), 1e-12)
        if rel > worst_err:
            worst_col, worst_err = int(col), float(rel)

    assert worst_err < 1e-3, (
        f"analytic jacobian disagrees with finite differences by "
        f"{worst_err:.3e} (relative) at column {worst_col}"
    )


@pytest.mark.data
@pytest.mark.slow
def test_codegen_is_reproducible(data_dir):
    """Regenerating the templates must produce byte-identical source.

    Codegen used to probe with the global numpy RNG, so the emitted source --
    and with it the jacobian's sparsity -- varied per process.
    """
    import pyCamSet.optimisation as optimisation

    template_dir = Path(optimisation.__file__).parent / "template_functions"
    names = ["jac", "loss", "matflow_jac"]

    def hashes() -> dict[str, str]:
        out = {}
        for path in sorted(template_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            if any(path.name.startswith(n) for n in names):
                out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return out

    for path in list(template_dir.glob("*.py")):
        if path.name != "__init__.py":
            path.unlink()
    np.random.seed(1)
    _build_ccube_problem(data_dir)
    first = hashes()

    for path in list(template_dir.glob("*.py")):
        if path.name != "__init__.py":
            path.unlink()
    np.random.seed(999)  # a different caller RNG state must not matter
    _build_ccube_problem(data_dir)
    second = hashes()

    assert first, "no templates were generated"
    assert first == second, (
        "regenerating the jacobian templates produced different source:\n"
        + "\n".join(
            f"  {name}: {first.get(name, '<missing>')[:16]} -> {second.get(name, '<missing>')[:16]}"
            for name in sorted(set(first) | set(second))
            if first.get(name) != second.get(name)
        )
    )
