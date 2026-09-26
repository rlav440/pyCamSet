# pcube Phase 1/2 full-corpus evidence

Status: executed on 2026-09-23 in `calibration_15092026`, with the pyCamSet-pcube worktree explicitly first on `sys.path`.

This is an evidence summary, not a claim that every supplied dataset supports a scientifically valid intrinsic solve. The complete machine-readable outputs remain outside the repository at:

`evidence/`

Important files:

- `full_corpus_phase1_phase2_evidence.json` — all eight Phase 1/2 variants and the frozen comparator.
- `quality_gate_report.json` — quality gates defined before result review and their per-variant result.
- `source_manifest_readback.json` — before/after hash and size comparison for every source-root file.
- `*_source_manifest.json` — pre-run manifests for each source root.

## Source coverage and immutability

| Dataset | Target | TIFFs | Input dtype/shape | Source files changed |
|---|---:|---:|---|---:|
| `pcube_4_squares` | 4 squares, 30 mm | 800 | uint16, 720x540 | 0 |
| `pcube_5_squares` | 5 squares, 30 mm | 800 | uint16, 720x540 | 0 |
| `rpan_6_squares` | 6 squares, 30 mm | 400 | uint8, four native crop shapes | 0 |
| `rpan_7_squares` | 7 squares, 30 mm | 400 | uint8, four native crop shapes | 0 |
| **Total** |  | **2,400** |  | **0** |

The source-manifest read-back covered all files, not just TIFFs: 807, 817, 401 and 401 files respectively were unchanged in size and SHA-256. No cache, workspace, calibration or rendered output was written under a source root.

The enabled one-pass preprocessing was compared against an independent frozen implementation on all 2,400 TIFFs. Mismatches: 0; maximum absolute pixel difference: 0 for every dataset.

## Phase results

Each row used `caching=False`, `threads=1`, all cameras, all images, a `PuzzleBoardCube` target, and a separate scratch workspace. Native means no rescale/gamma. Enabled means upper-byte conversion for uint16, gamma 0.5, one area resize at scale 0.25, with detections mapped back to native coordinates.

| Dataset | Mode | Phase 1 status | Detection rate range (mean) | Phase 2 disposition |
|---|---|---|---:|---|
| pcube 4 squares | native | **incomplete** | 0.00–0.00 (0.00) | blocked: no usable detections |
| pcube 4 squares | enabled | complete | 0.24–0.61 (0.445) | limitation: Zhang absolute conic not positive definite |
| pcube 5 squares | native | **incomplete** | 0.00–0.00 (0.00) | blocked: no usable detections |
| pcube 5 squares | enabled | complete | 0.51–0.93 (0.776) | limitation: Zhang absolute conic not positive definite |
| R_pan 6 squares | native | complete | 0.09–0.58 (0.395) | limitation: Zhang absolute conic not positive definite |
| R_pan 6 squares | enabled | complete | 0.48–0.69 (0.633) | limitation: Zhang absolute conic not positive definite |
| R_pan 7 squares | native | complete | 0.51–0.70 (0.620) | limitation: Zhang absolute conic not positive definite |
| R_pan 7 squares | enabled | complete | 0.33–0.50 (0.450) | **quality gates passed** |

The native pcube rows are deliberately recorded as `incomplete`, not green Phase 1 runs: all cameras saw zero target features. The Phase 1 workflow now persists `status=failed` for exceptions, `status=incomplete` for blocking detection reports, and `status=complete` only when no blocking flags exist. This also prevents a terminal message from calling a blocking detection run complete.

For R_pan 7 squares enabled, Phase 2 saved and reloaded an initial camset for all four cameras. Usable calibrated-view counts were 50, 47, 50 and 33; reprojection RMS median was 0.583 px and p95 was 0.818 px. Intrinsics and distortion were finite, principal points were inside native image bounds, native resolutions were preserved, and the save/load intrinsic round trip matched.

The other seven Phase 2 outcomes are dataset limitations under the stated target and lens model, not fabricated calibrations. The repeated Zhang diagnostic says the supplied views do not provide the distinct orientations required by the current pinhole seed. The enabled preprocessing materially improves detection on the pcube roots and on R_pan 6, but it cannot create missing pose diversity. No r_nebula intensity or metric accuracy claim is used as ground truth.

## Quality gates used before inspecting results

The machine-readable report records these gates explicitly:

1. source-root file hashes and sizes unchanged;
2. frozen preprocessing exact equality for every TIFF;
3. Phase 1 complete with no blocking flags;
4. Phase 2 has no error and an initial camset artifact;
5. no camera silently dropped between detection and calibration;
6. finite/plausible intrinsics and distortion, positive focal lengths, principal point in bounds, and focal length below 25 times the largest native dimension as a sanity bound;
7. at least ten detected views per camera;
8. finite detected-view residuals, median at most 2 px and p95 at most 5 px;
9. native image-size/resolution convention stable;
10. camset save/load preserves camera names and intrinsic values;
11. enabled-vs-native regression is checked where native Phase 2 succeeds, otherwise marked N/A rather than passed by assumption.

## Verification

- Windows symlink-privilege repair: the two previously failing real E2E tests passed after fixture camera trees were copied with `shutil.copytree`; no Developer Mode/admin privilege is required.
- Focused backend/workflow/quality suite: 85 passed.
- GUI phase-contract suite, headless (`QT_QPA_PLATFORM=offscreen`, `PYVISTA_OFF_SCREEN=true`, `MPLBACKEND=Agg`): 95 passed.
- `py_compile` for changed workflow, GUI and target modules/tests: passed.
- `git diff --check`: passed.
