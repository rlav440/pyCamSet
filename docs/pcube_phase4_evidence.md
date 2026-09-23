'''Purpose: Evidence report for the pcube Phase 4 self-calibration vertical slice.
Status: Active; records the fail-closed disposition of the real Phase 4 run.
Future: Add the four-dataset Phase 4 campaign matrix and a quality-gate-passing run when the upstream observation set supports it.
'''

# pcube Phase 4 evidence

This report records two real Phase 4 runs from verified Phase 3 outputs. It is
an evidence record, not a claim that either Phase 4 result is scientifically
accepted.

## Run identity

- Phase 3 input run: `20260923_014000_be0d2f`
- Phase 4 run: `20260923_032323_fb3dce`
- Source image root: `E:/R_pan/1 Data/2026-07-31/cut_tiffs_14h-14m-02s`
- Machine-readable output: `D:/Hermes/profiles/rebels/cache/scratch/pcube_p16x2_evidence/phase4_real_rpan7.json`
- Reloaded CameraSet: `D:/Hermes/profiles/rebels/cache/scratch/pcube_p16x2_evidence/phase3_real/.pycamset_workspace/phase4_runs/20260923_032323_fb3dce/self_calibrated_cameras.camset`

## Observed result

The backend completed and saved/reloaded a four-camera Phase 4 result, but the
quality gate correctly classified it as `incomplete`:

- initial mean reprojection error: 4.480463 px
- final mean reprojection error: 2.925415 px
- reprojection objective cost: 609596.8632 -> 104142.5216
- improvement against the Phase 3 final: 1.555048 px
- free target points: 205
- gauge-fixed target points: 3 (`[0, 1, 7]`)
- observed cameras: 4/4
- observed image indices: 75 of 92
- explicitly missing image indices: 17
- per-camera final means: view1 2.667614 px, view2 2.345418 px,
  view3 3.461900 px, view4 3.251131 px
- camera parameter drift: zero for the saved camera arrays in this run
- finite camera parameters and proper rotations: true for all four cameras

The blocking disposition is:

1. the SciPy solver reached `max_iter` and reported unsuccessful termination;
2. the observation graph has missing global image indices.

The result is therefore not presented as a successful self-calibration. The
Phase 4 GUI exposes the same disposition, blocking flags, camera/image coverage,
gauge accounting, and per-image residual count instead of equating a reduced
error with acceptance.

## Independent telecentric run

The M_NEBULA five-square telecentric Phase 3 output was also run without
changing the source images:

- Phase 3 input run: `20260923_021835_70e2cc`
- Phase 4 run: `20260923_031443_d5fb27`
- machine-readable output: `D:/Hermes/profiles/rebels/cache/scratch/pcube_p16x2_evidence/phase4_real_pcube5_telecentric.json`
- reloaded CameraSet: `D:/Hermes/profiles/rebels/cache/scratch/pcube_p16x2_evidence/phase3_telecentric_pcube5/.pycamset_workspace/phase4_runs/20260923_031443_d5fb27/self_calibrated_cameras.camset`
- observed cameras: 8/8
- observed images: 100, with no missing image indices
- initial mean reprojection error: 12.734540 px
- final mean reprojection error: 13.557749 px
- reprojection objective cost: 10264690.6723 -> 8528445.0210
- free target points: 150; gauge-fixed points: 3 (`[0, 1, 5]`)

Its only blocking flag was `final reprojection error did not improve finitely`.
This is the expected fail-closed outcome for a run that made the result worse:
the presence of complete observation coverage does not turn non-improvement
into a calibration success. The least-squares objective cost did decrease, but
the mean reprojection error and median error increased; the quality gate keeps
those distinct rather than treating objective reduction as scientific validity.
A repeat with `max_nfev=100` produced the same
12.734540 -> 13.557749 px result as `max_nfev=300`; the reported camera
intrinsic, distortion and extrinsic drift was zero for all eight cameras.
That controlled repeat is evidence that simply allowing more iterations does
not recover a useful Phase 4 solution for this telecentric corpus; it is not
presented as an underdetermination theorem.

The remaining physical limitation is explicit in the current pyCamSet model:
`pyCamSet/optimisation/function_block_implementations.py` defines the
telecentric extrinsic block as rotation-only because axial translation is an
exact gauge freedom, while `telecentric_intrinsic` carries magnification and
telecentricity. The Phase 4 gauge report therefore records the three fixed
target points and reference pose; it does not invent an unobservable
telecentric camera translation. This is a model-level limitation, not a
silent camera or image drop.

## Code and regression coverage

The implementation adds:

- fail-closed Phase 4 metadata status and persisted quality-gate disposition;
- explicit solver, finite-value, camera/image coverage, gauge, save/reload and
  camera-geometry checks, including separate objective-cost and mean-error
  dispositions;
- per-camera and per-image final residual diagnostics;
- fallback extraction of initial per-image errors when a real handler exposes an
  empty cache;
- a target-point-data-unit-aware self-calibration gauge spacing;
- GUI run/cancel/retry status handling and quality-gate presentation.

The missing-image gate and initial-error fallback each have regression tests.

## Verification

- Phase 4 contract: 8 passed.
- Workflow backend seam and phase tests plus Phase 4 contract: 153 passed.
- GUI phase contracts (offscreen): 95 passed.
- Bundle-handler tests: 43 passed, 13 pre-existing numerical/plot warnings.
- `py_compile`: passed for all changed Python files.
- `git diff --check`: passed.
- Mutation test: solver-success guard killed; missing-image guard killed; restore
  verified by the mutation harness.
- Real r_nebula runner: exit 0; quality disposition `incomplete` as reported above.
- Real M_NEBULA telecentric runner: exit 0 at `max_nfev=100`; quality disposition
  `incomplete` because the final mean error increased despite objective-cost
  reduction. The independent `max_nfev=300` repeat had the same metrics.

The source image roots were not modified. Bulk run artefacts remain outside the
repository.
