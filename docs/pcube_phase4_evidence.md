'''Purpose: Evidence report for the pcube Phase 4 self-calibration vertical slice.
Status: Active; records fail-closed and quality-gate-passing real Phase 4 runs.
Future: Add the four-dataset Phase 4 campaign matrix and native/offscreen visual evidence for the robust-loss run.
'''

# pcube Phase 4 evidence

This report records two real Phase 4 runs from verified Phase 3 outputs. It is
an evidence record, not a claim that either Phase 4 result is scientifically
accepted.

## Run identity

- Phase 3 input run: `20260923_014000_be0d2f`
- Phase 4 run: `20260923_032323_fb3dce`
- Source image root: `the R_pan 7-square dataset root`
- Machine-readable output: `evidence/phase4_real_rpan7.json`
- Reloaded CameraSet: `evidence/phase3_real/.pycamset_workspace/phase4_runs/20260923_032323_fb3dce/self_calibrated_cameras.camset`

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
- pre-fix metadata recorded zero camera drift, but that value is invalid because
  the Phase 4 output path mutated the Phase 3 Camera objects before comparison;
  this incomplete run is not used as drift evidence
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
- machine-readable output: `evidence/phase4_real_pcube5_telecentric.json`
- reloaded CameraSet: `evidence/phase3_telecentric_pcube5/.pycamset_workspace/phase4_runs/20260923_031443_d5fb27/self_calibrated_cameras.camset`
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
12.734540 -> 13.557749 px result as `max_nfev=300`; its pre-fix zero-drift
metadata is not trusted for the same aliasing reason described above.
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

## Controlled parameter-lock probes

The Phase 4 entry point also had a real fixed-parameter warm-start defect:
`set_from_templated_camset` copied the longer Phase 3 flat vector directly into
the shorter Phase 4 vector. The implementation now rehydrates the primitive
arrays and repacks only the parameters that remain free; this is covered by a
regression test. The fix was exercised against the same eight-camera
telecentric Phase 3 artefact rather than only a synthetic handler.

Three controlled probes then tested whether pinning the telecentric camera
extrinsics, intrinsics, or both could produce a good result:

| fixed parameters | final mean error (px) | objective cost | solver |
| --- | ---: | ---: | --- |
| extrinsics | 13.539554 | 8,603,475.4008 | successful, 16 evaluations |
| intrinsics | 13.639066 | 8,817,958.2747 | successful, 9 evaluations |
| both | 13.433844 | 8,936,825.8943 | successful, 8 evaluations |

All three start at 12.734540 px and reduce the objective while worsening the
mean error. The complete machine-readable probe output is
`evidence/phase4_telecentric_fixed_variants.json`.
This rules out the simplest camera-parameter warm-start explanation without
weakening the quality gate. Together with the repeated all-free run and the
rotation-only telecentric extrinsic model, the current evidence supports a
model/data-level limitation for this telecentric corpus rather than a reason to
accept its Phase 4 result.

## GOOD robust-loss Phase 4 run

The rejection above exposed a real implementation defect: the GUI/API exposed
`loss` and `f_scale`, but the custom Schur path silently ignored them. The
backend now routes non-linear losses through SciPy's trust-region solver and
passes the requested `loss` and `f_scale` values. The validated telecentric
recipe (`loss=soft_l1`, `f_scale=1.0`, `max_nfev=100`) was then executed through
the actual `phase4.run` workflow:

- run: `20260923_040725_7a42b3`
- deterministic repeat: `20260923_040751_fba3d2`
- machine-readable output: `evidence/phase4_real_pcube5_telecentric.json`
- first-run preserved copy: `evidence/phase4_real_pcube5_telecentric_soft_l1_first.json`
- save/reload artefacts: both run directories contain `self_calibrated_cameras.camset`
- solver: successful trust-region termination; 100 image indices observed for all 8 cameras
- mean reprojection error: `12.734540 -> 5.832703 px`
- Phase 3 comparison: `12.734540 -> 5.832703 px` (improvement `6.901837 px`)
- target shape displacement: `0.684601 mm` mean; gauge scale factor `0.976606`
- camera quality: all eight finite, positive-focal, proper-rotation checks passed
- recomputed maximum absolute camera-parameter change against the saved Phase 3
  CameraSet: intrinsic `522.7668429102`, distortion `1.6208375997`, extrinsic
  `0.0517495545` (both GOOD runs); the per-camera normalized drift is now
  persisted in the machine-readable metadata rather than reported as zero
- quality-gate status: `complete`, with no blocking flags
- repeat spread: `0.0 px` final mean and `0.0` objective-cost difference across the two runs

The robust run's raw least-squares reprojection cost is higher than the linear
run's cost because `soft_l1` deliberately optimises a different robust
objective. That is recorded as `objective_cost_reduced=false`; it does not
invalidate the run because the acceptance gate is based on finite, successful,
fully covered, physically plausible calibration with reduced native-pixel
reprojection error. The result is the required GOOD Phase 4 calibration, while
the earlier linear result remains correctly classified as incomplete.

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
- non-linear loss handling that cannot silently fall through the custom Schur
  path; SciPy trust-region receives the requested loss and scale.

The missing-image gate and initial-error fallback each have regression tests.

## Verification

- Phase 4 contract: 9 passed.
- Workflow backend seam and phase tests plus Phase 4 contract: 155 passed.
- GUI phase contracts (offscreen): 95 passed.
- Bundle-handler tests: 43 passed, 13 pre-existing numerical/plot warnings.
- `py_compile`: passed for all changed Python files.
- `git diff --check`: passed.
- Mutation test: solver-success guard killed; missing-image guard killed; restore
  verified by the mutation harness.
- Fixed-camera warm-start regression: passed; the real telecentric lock probes
  completed for extrinsics, intrinsics, and both together.
- Camera-drift regression: passed; Phase 4 now clones geometry-only Camera
  objects, uses `set_extrinsic`, and preserves the Phase 3 rig for comparison.
- Saved GOOD-run drift recomputation: passed; both runs produce the same camera
  parameter arrays, while their serialised camset files have distinct hashes.
- Robust-loss solver routing regression: passed; the custom Schur path is not
  used when `loss` is non-linear.
- Real r_nebula runner: exit 0; quality disposition `incomplete` as reported above.
- Real M_NEBULA telecentric runner: exit 0 at `max_nfev=100`; quality disposition
  `incomplete` because the final mean error increased despite objective-cost
  reduction. The independent `max_nfev=300` repeat had the same metrics.
- Real M_NEBULA telecentric robust-loss runner: exit 0 twice; quality disposition
  `complete` with exact repeat metrics and save/reload artefacts as recorded
  above.

The source image roots were not modified. Bulk run artefacts remain outside the
repository.

## GUI and upstream-boundary proof

The real `Phase4Tab` and `Phase4DiagnosticsTab` were instantiated against the
saved M_NEBULA Phase 3/Phase 4 workspace under
`QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true`. The diagnostics tab loaded
the exact GOOD repeat `20260923_040751_fba3d2`, rendered its summary, and the
PyVista export path produced a real 3D camera/target PNG:

- GUI diagnostics summary: `evidence/phase4_gui_summary_good.png`
- GUI Phase 4 settings: `evidence/phase4_gui_settings_good.png`
- PyVista 3D render: `evidence/phase4_pyvista_good_render.png`
- GUI state evidence: `evidence/phase4_gui_state_evidence.json`

The state harness exercised the actual tab callbacks and recorded the honest
labels `Cancellation requested; waiting for the active solver step…`,
`Incomplete — quality gate blocked hand-off: ...`, `Failed — demo failure`, and
`Complete — quality gate passed`; retry was disabled after the complete result.
This supplements the existing 95 headless GUI contract tests with a real saved
run, diagnostics rendering, visual export, and state-transition evidence.

Phase 4 was deliberately not fabricated for the two R_pan cut-TIFF roots that
did not produce a valid Phase 3 input. The upstream four-dataset record
(`docs/pcube_phase3_evidence.md`, lines 67–85) records both r_nebula 6-squares
and the native 7-squares dispositions as upstream-limited, while the enabled
7-squares root produced the verified pinhole Phase 3 run used above. The same
record covers the M_NEBULA 4-squares native/enabled failures and the
M_NEBULA 5-squares native zero-detection path. Thus the Phase 4 GUI/backend
uses the exact provenance-matching Phase 3 output where one exists and reports
upstream failure rather than silently changing roots, preprocessing, or models.
