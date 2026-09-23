'''Purpose: Evidence record for the pcube Phase 3 vertical slice.
Status: Active; generated from the verified P16X-2 corpus and a real Phase 3 run.
Future: Re-run the four-dataset matrix when an upstream Phase 2 limitation changes.
'''

# pcube Phase 3 end-to-end evidence

## Scope

This record covers the Phase 3 backend and GUI seam after the P16X-2 Phase 1/2
corpus run. Source TIFF roots were treated as read-only. Bulk run directories
remain outside git under
`D:/Hermes/profiles/rebels/cache/scratch/pcube_p16x2_evidence/`.

## Backend result

[VERIFIED-BY-EXECUTION] The real Phase 3 runner consumed the P16X-2
`rpan_7_squares/enabled` Phase 1 detection pickle and Phase 2 camset, then
saved and reloaded:

- run: `20260923_014000_be0d2f`
- disposition: `complete`
- solver: SciPy trust-region reflective, status 3 (`xtol` termination)
- observations: 6,759 across all four cameras
- parameters: finite; camera observation graph covered indices 0, 1, 2, 3
- mean Euclidean reprojection: 20.3508 px initial -> 4.4805 px final
- reduction ratio: 4.5421x
- per-camera means: view1 2.9002 px, view2 1.7608 px, view3 3.6291 px,
  view4 11.4842 px
- save/reload: `load_CameraSet` recovered four cameras and a calibration report

The per-camera values are retained as a warning signal even though the global
quality gate passed; view4 is materially worse than the other cameras and must
not be hidden by the aggregate.

[VERIFIED-BY-EXECUTION] A controlled rerun with `max_nfev=100` reduced the
error to 4.4242 px but terminated with `success=false` (`maximum number of
function evaluations is exceeded`) and was recorded as `incomplete`, not as a
good calibration. This is the intended fail-closed disposition.

## Quality gate

The backend now records a machine-readable `diagnostics.quality_gate` and a
run-level `status`:

- `failed`: the phase raised while loading or solving;
- `incomplete`: a result was saved for diagnosis, but a quality gate blocked
  hand-off;
- `complete`: solver success, finite parameters/residuals, observations,
  camera coverage, and strict error reduction all hold.

The GUI consumes the same blocking reasons when deciding whether Phase 4 may
be continued. It displays the disposition and quality-gate flags in the Phase
3 diagnostics summary rather than treating an optimiser exit as proof of a
good calibration.

## Four-dataset disposition

[VERIFIED-BY-EXECUTION] These are the P16X-2 full-corpus outcomes; Phase 3 was
only run where Phase 2 produced a valid camset.

| Dataset / preprocessing | Phase 2 disposition | Phase 3 disposition |
|---|---|---|
| M_NEBULA pcube 4 squares / native | zero detections | not run; no valid input |
| M_NEBULA pcube 4 squares / enabled | Zhang pose-coverage failure | not run; no valid input |
| M_NEBULA pcube 5 squares / native | zero detections | not run; no valid input |
| M_NEBULA pcube 5 squares / enabled | Zhang pose-coverage failure | not run; no valid input |
| r_nebula 6 squares / native | Zhang pose-coverage failure | not run; no valid input |
| r_nebula 6 squares / enabled | Zhang pose-coverage failure | not run; no valid input |
| r_nebula 7 squares / native | Zhang pose-coverage failure | not run; no valid input |
| r_nebula 7 squares / enabled | valid Phase 2 camset | real Phase 3 run above |

No Phase 3 result was fabricated for an upstream-limited dataset, and no
source image tree was modified. The enabled/native detection comparison is in
`docs/pcube_phase1_phase2_evidence.md` and the machine-readable P16X-2 corpus
artefacts.

## GUI proof

[VERIFIED-BY-EXECUTION] The real `Phase3DiagnosticsTab` was instantiated under
`QT_QPA_PLATFORM=offscreen`, loaded the saved Phase 3 workspace, rendered its
summary/diagnostics tabs, and captured:

`D:/Hermes/profiles/rebels/cache/scratch/pcube_p16x2_evidence/phase3_real/phase3_diagnostics_offscreen.png`

The diagnostics data also exposes the initial per-image values, residual XY
scatter, per-camera means and the camera-pose view hook used by the GUI.

## Tests

[VERIFIED-BY-EXECUTION]

- targeted Phase 3 quality contract: 3 passed;
- workflow/backend seam plus workflow phase tests: 148 passed;
- headless GUI phase contracts: 95 passed;
- combined focused regression command: 243 passed;
- `py_compile` passed for all changed Python modules and the new test.

The broad repository suite did not complete within the bounded run: it entered
existing long-running ArUco corpus tests and was terminated by the timeout.
That result is inconclusive, not a pass claim.
