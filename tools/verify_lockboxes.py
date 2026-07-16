'''
Purpose: Verify Phase 3 Gaussian camera lockbox backend wiring.
Status:  Targeted smoke checks for the extrinsic-parameter MVP.
Future:  Extend with full bundle-adjustment fixture coverage after GUI workflow wiring.
'''

from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyCamSet.optimisation.camera_lockbox import (  # noqa: E402
    CameraLockboxConfig,
    append_lockbox_jacobian,
    append_lockbox_residuals,
    apply_lockbox_bounds,
    build_extrinsic_parameter_lockbox,
    make_disabled_prior,
)


def check(number: int, message: str) -> None:
    print(f"CHECK {number}/10 {message}")


def assert_all_inf_bounds(bounds: tuple[np.ndarray, np.ndarray]) -> None:
    lower, upper = bounds
    assert np.all(np.isneginf(lower)), lower
    assert np.all(np.isposinf(upper)), upper


def main() -> None:
    param_len = 60
    intr_end = 18
    cam_names = ["cam0", "cam1", "cam2"]
    source = np.array(
        [
            [0.10, 0.20, 0.30, 1.0, 2.0, 3.0],
            [0.40, 0.50, 0.60, 4.0, 5.0, 6.0],
            [0.70, 0.80, 0.90, 7.0, 8.0, 9.0],
        ],
        dtype=float,
    )

    disabled = make_disabled_prior(param_len)
    assert disabled.indices.size == 0
    assert_all_inf_bounds(apply_lockbox_bounds(param_len, disabled))
    check(1, "disabled config gives infinite bounds")

    config = CameraLockboxConfig(
        enabled=True,
        rotation_half_width=0.01,
        translation_half_width=0.25,
        rotation_sigma=0.005,
        translation_sigma=0.10,
    )
    prior = build_extrinsic_parameter_lockbox(
        cam_names=cam_names,
        extr_unfixed=np.array([True, False, True]),
        source_extrinsics=source,
        intr_end=intr_end,
        param_len=param_len,
        config=config,
    )
    assert set(prior.camera_names) == {"cam0", "cam2"}
    assert "cam1" not in prior.camera_names
    check(2, "fixed cameras are skipped")

    expected_indices = np.array(list(range(18, 24)) + list(range(24, 30)), dtype=int)
    assert np.array_equal(prior.indices, expected_indices), prior.indices
    assert prior.indices[0] == intr_end
    assert 30 not in prior.indices  # raw camera index 2 times six plus intr_end would land here.
    check(3, "packed extrinsic indices include intr_end and avoid raw camera_index times six")

    lower, upper = apply_lockbox_bounds(param_len, prior)
    assert lower.shape == (param_len,)
    assert upper.shape == (param_len,)
    assert np.isclose(lower[18], source[0, 0] - config.rotation_half_width)
    assert np.isclose(upper[23], source[0, 5] + config.translation_half_width)
    assert np.isneginf(lower[30]) and np.isposinf(upper[30])
    check(4, "bounds arrays have exact param_len and finite constrained entries only")

    params = np.zeros(param_len, dtype=float)
    params[prior.indices] = prior.centres + prior.sigmas
    base_residuals = np.array([11.0, 12.0], dtype=float)
    residuals = append_lockbox_residuals(base_residuals, params, prior)
    assert residuals.shape == (base_residuals.size + prior.indices.size,)
    assert np.allclose(residuals[-prior.indices.size:], 1.0)
    check(5, "residual append count and values are correct")

    dense_base = np.zeros((3, param_len), dtype=float)
    dense_jac = append_lockbox_jacobian(dense_base, param_len, prior)
    assert dense_jac.shape == (3 + prior.indices.size, param_len)
    assert np.isclose(dense_jac[3, prior.indices[0]], 1.0 / prior.sigmas[0])
    check(6, "dense Jacobian append shape is correct")

    sparse_base = sparse.csr_array((3, param_len), dtype=float)
    sparse_jac = append_lockbox_jacobian(sparse_base, param_len, prior)
    assert sparse.issparse(sparse_jac)
    assert sparse_jac.shape == (3 + prior.indices.size, param_len)
    assert np.isclose(sparse_jac[3, prior.indices[0]], 1.0 / prior.sigmas[0])
    check(7, "sparse Jacobian append works")

    changed_modules = [
        ROOT / "pyCamSet" / "optimisation" / "camera_lockbox.py",
        ROOT / "pyCamSet" / "optimisation" / "template_handler.py",
        ROOT / "pyCamSet" / "optimisation" / "optimisation_handling.py",
    ]
    for module in changed_modules:
        py_compile.compile(str(module), doraise=True)
    check(8, "changed Python modules compile")

    template_source = (ROOT / "pyCamSet" / "optimisation" / "template_handler.py").read_text(encoding="utf-8")
    template_tree = ast.parse(template_source)
    handler_classes = [node for node in template_tree.body if isinstance(node, ast.ClassDef) and node.name == "TemplateBundleHandler"]
    assert handler_classes, "TemplateBundleHandler class not found"
    method_names = {node.name for node in handler_classes[0].body if isinstance(node, ast.FunctionDef)}
    assert "get_lockbox_bounds" in method_names
    assert "_ensure_lockbox_prior" in method_names
    assert "cam_idx * 6" not in (ROOT / "pyCamSet" / "optimisation" / "camera_lockbox.py").read_text(encoding="utf-8")
    check(9, "TemplateBundleHandler exposes lockbox bounds and helper uses packed indices")

    opt_source = (ROOT / "pyCamSet" / "optimisation" / "optimisation_handling.py").read_text(encoding="utf-8")
    opt_tree = ast.parse(opt_source)
    least_squares_calls = [
        node for node in ast.walk(opt_tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "least_squares"
    ]
    assert len(least_squares_calls) == 2, len(least_squares_calls)
    assert all(any(keyword.arg == "bounds" for keyword in call.keywords) for call in least_squares_calls)
    assert opt_source.count("get_lockbox_bounds(len(init_params))") == 2
    check(10, "both least_squares calls include bounds from get_lockbox_bounds")


if __name__ == "__main__":
    main()
