# The workflow package

!!! warning "This is an internal tool"

    `pyCamSet.workflow` exists to drive the [graphical
    workflow](../how-to/gui.md), and is verbose because it has to support
    programmatic replay: every setting is data, every run is recorded, and
    nothing is implicit. That makes it a poor thing to write a script against.

    **For a script, use
    [`calibrate_cameras`](../how-to/calibrate.md) and, where the target's own
    geometry needs to move, [explicit
    self-calibration](../how-to/self-calibration.md).** Both do the same work in
    a fraction of the code, and both are the supported surface.

    This page is here so that the GUI's behaviour can be understood and
    reproduced, not as a recommended path.

---

`calibrate_cameras` does the whole job in one call. The workflow package is the
same job broken into four phases that can be run, inspected and re-run
individually — and it is what `pycamset` drives, so the GUI and a script do the
same work and write the same files.

Nothing in `pyCamSet.workflow` imports a GUI toolkit. It runs on a lean install,
and on a machine with no display.

The four phases are **detection**, **intrinsics**, **bundle adjustment** and
**self-calibration**, corresponding one-to-one with the GUI's Phase 1 through
Phase 4. Each is a function taking settings, a workspace, and somewhere to send
its output, returning the run record it saved.

## The workspace

A workspace is a directory beside the image folder that holds every run: its
settings, its diagnostics, and the files it produced. Runs are addressed by a
run id, and each records the runs it came from, so phase 3 can find the phase 2
that fed it.

```python
from pyCamSet.workflow import WorkspaceManager, workspace_path_for

workspace = WorkspaceManager(workspace_path_for('/data/my_rig'))
```

`load_runs` lists what a phase has produced, `find_run` fetches one by id, and
`build_predecessor_chain` walks a run back through the phases that produced its
inputs.

## Running a phase

Settings are a plain dictionary. The target is described by a spec rather than
constructed directly, so the same settings can be written to disk and replayed
— which is the verbosity this package exists to carry:

```python
from pyCamSet.workflow import WorkspaceManager, phase1, workspace_path_for

workspace = WorkspaceManager(workspace_path_for('/data/my_rig'))

params = {
    'target': {
        'type': 'ChArUco',
        'num_squares_x': 20,
        'num_squares_y': 20,
        'square_size': 4.0,
        'marker_fraction': 0.8,
        'marker_backend': 'aruco1',
        'a_dict': 3,
        'legacy': True,
    },
    'f_loc': '/data/my_rig',
    'caching': True,
    'high_distortion': False,
    'n_lim': None,
    'threads': 1,
    'upscale_factor': 1,
    'fixed_params': None,
    'problem_options': None,
    'selected_cameras': [],
}

run = phase1.run(params, workspace, log=print)
```

`log` is called with each line of output; it defaults to
`pyCamSet.workflow.discard`, so passing `print` is how a script sees progress,
and passing a list's `append` is how the GUI captures it.

## Reading the result

A phase returns the metadata record it saved. `run['error']` is `None` when the
phase succeeded, and the diagnostics sit under `run['diagnostics']`:

```python
if run['error'] is not None:
    raise RuntimeError(run['error'])

diagnostics = run['diagnostics']
print(diagnostics['cam_names'])
print(diagnostics['D1.2_detection_rate'])   # per camera, 0.0 to 1.0
print(diagnostics['D1.7_min_features'])
```

The files a phase wrote are under `run['artifacts']`, keyed by name, and can be
recovered later through the workspace:

```python
saved = workspace.find_run('phase1', run['run_id'])
pickle_path = saved['artifacts']['detected_datapoints_pickle']
```

!!! note

    Phase 1 only records its detections as an artifact when `caching` is `True`.
    Run it with caching off and the run is saved with its diagnostics but no
    `artifacts` entry, and the later phases have nothing to pick up.

## Chaining the phases

Later phases take the run they follow. Passed explicitly they use it; left out,
they find the most recent suitable run in the workspace:

```python
from pyCamSet.workflow import phase1, phase2, phase3, phase4

p1 = phase1.run(params, workspace, log=print)
p2 = phase2.run(params, workspace, log=print, phase1_run=p1)
p3 = phase3.run(params, workspace, log=print, phase1_run=p1, phase2_run=p2)
p4 = phase4.run(params, workspace, log=print, phase3_run=p3)
```

Invalid settings raise `pyCamSet.workflow.ParamError`, which carries a message
written to be shown to a user rather than a stack trace.

## Output and plotting

Phases that would otherwise draw to the screen can be made to behave in a script
or on a server with the helpers in `pyCamSet.workflow.logs`: `captured_output`
redirects what a phase prints into your `log`, and `non_interactive_plotting`
puts matplotlib on a non-interactive backend for the duration.
