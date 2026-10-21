# Testing the pipeline on simulated captures

A real capture tells you how well a calibration agrees with itself. A simulated
one also tells you how far it is from the cameras and the target that actually
made the images, because you made them. `pyCamSet.simulation` builds such
captures: a known camera rig photographs a known, slightly imperfect cube
target, through a realistic sensor. The images are written in the folder layout
the real pipeline reads, and the ground truth goes beside them.

This page covers what is simulated and why, how to run the whole pipeline on a
simulation and score it, and how to extend the simulation to other targets,
rigs and imperfections.

!!! note "What this does and does not prove"

    A simulation tests the pipeline against the model the simulation is built
    on. It catches detector failures, solver problems and unit mistakes, and it
    measures how errors in the target propagate. It does not prove that real
    photographs behave the same way: real lenses, lighting, paper and printers
    have effects the model leaves out (listed under
    [Extending it](#extending-it)).

---

## Quick start

From the repository root, with the environment pyCamSet is installed in:

```bash
python scripts/simulate_calibration.py --out sim_runs --threads 4 \
    --target '{"type": "Ccube", "length": 80, "n_points": 6}' \
    --target '{"type": "Ccube2", "length": 80, "n_points": 6}'
```

For each target this renders 4 cameras x 30 shots, runs Phases 1-4 exactly as
the GUI does, and writes `sim_runs/report.md` and `sim_runs/report.json`. Each
target's images end up in `sim_runs/<target>/images/`, with the Phase 1-4 runs
in that folder's `.pycamset_workspace`, so you can open the folder in the GUI and
step through every phase in the education windows.

Useful options:

| Option | Default | |
|---|---|---|
| `--cameras` | 4 | Cameras on the ring |
| `--shots` | 30 | Target poses, each photographed by every camera |
| `--radius` | 0.35 | Ring radius, metres |
| `--focal` | 1800 | Nominal focal length, pixels |
| `--res` | 1600 1200 | Image width and height |
| `--wander` | 0.06 | How far the target moves from the centre, metres, so it reaches the image edges |
| `--perfect` | off | Paste the faces perfectly instead of by hand |
| `--threads` | 1 | Detection processes |
| `--seed` | 0 | Everything random is seeded from this |
| `--rescore` | off | Score the captures already in `--out` again, without rendering or calibrating |

Rendering and detection take minutes per target at the defaults; a smaller
run (for example `--cameras 3 --shots 8`) finishes much sooner.

---

## What is simulated

Each part of the imaging chain is one module, so you can replace one part
without touching the others.

### The rig: `simulation.rig`

`ring_rig()` places cameras on a ring around the origin, all looking at it,
alternately above and below the ring's plane: the inward-facing rig a cube
target is made for. Each camera is a [`SimCamera`][pyCamSet.simulation.rig.SimCamera]
with an OpenCV camera matrix, five distortion coefficients (`k1, k2, p1, p2, k3`,
the model pyCamSet itself uses), a resolution and a world-to-camera extrinsic.

Cameras of one model are never identical, so each one's focal length, principal
point and distortion are drawn within `spread` (3 %) of the nominal values. The
defaults are a 1600 x 1200 camera with an 1800 px focal length, mild barrel
distortion (`k1 = -0.08`, `k2 = 0.03`) and a ring 0.35 m across.

### The shots: `simulation.poses`

`held_target_poses()` turns the cube through uniformly random orientations, as a
person turns it in front of the rig, and moves it up to `translation` metres from
the centre on each axis. Moving it matters: a target that stays in the middle of
every image leaves the lens edges unmeasured, and the distortion there is then
a guess.

### The target as made: `simulation.imperfection`

A printed net pasted onto a 3D-printed core by hand is never placed perfectly.
[`PastingErrors`][pyCamSet.simulation.imperfection.PastingErrors] holds, per
face:

- a **rotation** of the paper about the face's centre (standard deviation 0.5°);
- an **offset** within the face (0.3 mm per axis);
- a **scale**: one printer scale error shared by the whole net (up to ±0.2 %),
  plus each face's own paper stretch per axis (0.1 %).

`paste_faces()` returns where the target's points really are. The calibration is
still told the nominal `point_data`, as it would be in the lab, so the
difference shows up as residual until Phase 4 refines the target's shape. How far
that is depends on the target's size, because the rotation and stretch move
points further on a bigger face. With the defaults and seed 2 (what the script
uses for `--seed 0`), a Ccube of `length` 80 and `n_points` 6 is pasted about
0.37 mm RMS from nominal (a Ccube2 of the same size, 0.46 mm), after the best
rigid fit; the report gives the figure for your own run as `pasting_rms_mm`.
Where the
paper overhangs a face's edge it is trimmed, and where it falls short the
plastic core shows.

### The picture: `simulation.render`

`render_view()` draws each face that turns towards the camera as two layers: the
core's face in plastic grey, and the paper on it (the target's own face texture,
moved by the pasting errors and trimmed to the face). One distant light shades
each face by its angle, and a softly mottled background fills the rest. The
picture is drawn three times larger in each direction and averaged down, so
edges are anti-aliased the way a sensor's pixels average light.

The renderer works for any cube target with face `textures`, `faceData` and a
`length` in metres: Ccube and Ccube2. PuzzleBoardCube has no face textures
yet, and the planar and icosahedral targets need a different face mapping.

### The camera: `simulation.sensor`

[`SensorModel`][pyCamSet.simulation.sensor.SensorModel] turns the ideal picture
into what a camera records, in this order:

1. **Lens distortion**: each output pixel is traced back through the inverse of
   the camera's distortion, which is inverted numerically to well under a
   thousandth of a pixel.
2. **Blur**: a Gaussian of 0.8 px, for lens softness and slight defocus.
3. **Vignetting**: the corners are 25 % darker than the centre.
4. **Noise**: shot noise (Poisson, 10,000 electrons full well, white paper at
   70 %) and read noise (8 electrons).
5. **Encoding**: gamma 2.2, then 8 bits.

---

## The ground truth

`simulate_capture()` writes, beside the camera folders:

- `ground_truth.json`: the target spec, every camera (intrinsics, distortion,
  resolution, extrinsic), every pose, the pasting errors, and the sensor and
  scene settings;
- `ground_truth_points.npy`: the true (pasted) target points, shaped like the
  target's `point_data`.

`load_ground_truth(folder)` reads both back as a
[`Capture`][pyCamSet.simulation.capture.Capture].

!!! note "Pixel centres"

    The ground truth uses OpenCV's projection convention, in which a pixel's
    centre is at integer coordinates. Every pyCamSet detector returns points in
    the same pixel-centre frame, under OpenCV 4 and 5 alike, so a calibration
    from their detections is compared with the true principal point directly.

---

## Scoring a calibration

[`compare_to_truth(cams, capture, solved_points=None)`][pyCamSet.simulation.evaluate.compare_to_truth]
reports, per camera:

- `focal_error_px` and `focal_error_pct`: estimated minus true focal length;
- `principal_point_error_px`;
- `k1_error`, `k2_error`: coefficient errors. These trade off against each other
  and against the focal length, so read them together with the next two;
- `lens_rms_px`, `lens_p99_px`: how far apart the true and estimated lenses put
  the same rays, over the image positions the target was actually seen at. This
  is the lens error that matters, measured only where there was data;
- `position_error_mm`, `rotation_error_deg`: each camera's pose relative to the
  first camera, against the truth. A calibration's world frame is its own
  choice, so poses are compared camera to camera. Phase 2 does not solve
  poses, so they are not scored there.

It also reports the reprojection RMS, `pasting_rms_mm` (how far the true points
are from nominal), and, given Phase 4's solved points, `target_shape_rms_mm`: how
far the refined target is from the true one, after the best rigid fit.

---

## Looking at a simulated run in the education windows

The script runs the phases through the same workflow modules the GUI uses, with
education recording on, so a simulated folder opens in the GUI like a real
one. Point the image folder at `<out>/<target>/images` and open the latest run
of each phase. Phase 1's "Finding markers" stage walks through the target's own
detector, and Phases 2-4 show the board views, the rig solve and the target
refinement as for a real capture. A simulation is a
good way to learn what each stage shows, because you know what the right
answer is.

---

## Extending it

The parts are independent, so most extensions touch one module:

- **Another cube target**: anything with `textures`, `faceData` and a metric
  `length` renders as is; pass its spec with `--target`. For PuzzleBoardCube, add
  face textures. For an icosahedron, generalise `render_view`'s face square to
  the face's own polygon (`faceData` already places any face).
- **Another rig**: write a function returning a list of `SimCamera`s, as
  `ring_rig` does: a stereo pair, a dome, cameras of different resolutions.
- **A telecentric camera**: `render_view` projects with a pinhole matrix; a
  telecentric camera needs an affine projection there and in `project_points`,
  and `lens_model="telecentric"` in the phase parameters.
- **Other target imperfections**: a bubble or a lifted corner moves points out of
  the face's plane. Add a field to `PastingErrors`, apply it in `paste_faces` and
  in `render_view`'s paper mapping, and the rest follows. A core printed slightly
  out of square would change `face_to_world`.
- **Other image effects**: motion blur, defocus that varies with depth,
  specular glare on glossy paper, JPEG compression, rolling shutter. Each is a
  step in `SensorModel.apply`, or in `render_view` for effects that depend on
  geometry.
- **A regression test**: `tests/test_simulation.py` checks the imaging chain in
  seconds. A slower check could run `simulate_one` on a small capture and assert
  on its scores.

Every random choice is seeded, so a capture can be rebuilt exactly from its
seed. Keep it that way when you add a part.
