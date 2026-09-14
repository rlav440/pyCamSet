================================================================================
The Calibration Workflow
================================================================================

Introduction
============

:func:`~pyCamSet.calibrate_cameras` does the whole job in one call. The workflow package
is the same job broken into four phases you can run, inspect and re-run
individually -- and it is what the ``pycamset`` GUI drives, so a script and the
GUI do the same work and write the same files.

Nothing in :mod:`pyCamSet.workflow` imports a GUI toolkit. It runs on a lean
install, and on a machine with no display.

The four phases are:

**Phase 1 -- detection**
   Reads an image folder, finds the calibration target in every image, and
   saves the detections plus the diagnostics that say whether they are worth
   calibrating from.

**Phase 2 -- intrinsics**
   Calibrates every camera independently from phase 1's detections, producing
   the camset that phase 3 starts from.

**Phase 3 -- bundle adjustment**
   Solves the whole camera set and the target poses together, optionally
   holding the cameras near a known rig geometry through the lockbox priors.

**Phase 4 -- self-calibration**
   Lets the target's own points become free parameters as well, gauged against
   a fixed subset so the solve cannot simply scale everything away.

Each phase is a function that takes settings, a workspace, and somewhere to
send its output, and returns the run record it saved.

The workspace
=============

A workspace is a directory beside the image folder that holds every run: its
settings, its diagnostics, and the files it produced. Runs are addressed by a
run id, and each one records the runs it came from, so phase 3 can find the
phase 2 that fed it.

.. code-block:: python

   from pyCamSet.workflow import WorkspaceManager, workspace_path_for

   workspace = WorkspaceManager(workspace_path_for('/data/my_rig'))

:meth:`~pyCamSet.workflow.WorkspaceManager.load_runs` lists what a phase has produced,
:meth:`~pyCamSet.workflow.WorkspaceManager.find_run` fetches one by id, and
:meth:`~pyCamSet.workflow.WorkspaceManager.build_predecessor_chain` walks a run back through the
phases that produced its inputs.

Running a phase
===============

Settings are a plain dictionary. The target is described by a spec rather than
constructed directly, so that the same settings can be written to disk and
replayed:

.. code-block:: python

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

``log`` is called with each line of output; it defaults to
:func:`pyCamSet.workflow.discard`, so passing ``print`` is how you see progress
in a script. Passing a list's ``append`` is how the GUI captures it.

Reading the result
==================

A phase returns the metadata record it saved. ``run['error']`` is ``None`` when
the phase succeeded, and the phase's diagnostics sit under
``run['diagnostics']``:

.. code-block:: python

   if run['error'] is not None:
       raise RuntimeError(run['error'])

   diagnostics = run['diagnostics']
   print(diagnostics['cam_names'])
   print(diagnostics['D1.2_detection_rate'])   # per camera, 0.0 to 1.0
   print(diagnostics['D1.7_min_features'])

The files a phase wrote are under ``run['artifacts']``, keyed by name, and can
be recovered later through the workspace:

.. code-block:: python

   saved = workspace.find_run('phase1', run['run_id'])
   pickle_path = saved['artifacts']['detected_datapoints_pickle']

.. note::

   Phase 1 only records its detections as an artifact when ``caching`` is
   ``True``. Run it with ``caching`` off and the run is saved with its
   diagnostics but no ``artifacts`` entry, and the later phases have nothing to
   pick up.

Chaining the phases
===================

Later phases take the run they follow. Passed explicitly, they use it; left
out, they find the most recent suitable run in the workspace:

.. code-block:: python

   from pyCamSet.workflow import phase1, phase2, phase3, phase4

   p1 = phase1.run(params, workspace, log=print)
   p2 = phase2.run(params, workspace, log=print, phase1_run=p1)
   p3 = phase3.run(params, workspace, log=print, phase1_run=p1, phase2_run=p2)
   p4 = phase4.run(params, workspace, log=print, phase3_run=p3)

Invalid settings raise :class:`pyCamSet.workflow.ParamError`, which carries a
message written to be shown to a user rather than a stack trace:

.. code-block:: python

   from pyCamSet.workflow import ParamError

   try:
       run = phase1.run(params, workspace, log=print)
   except ParamError as exc:
       print(f'Could not start: {exc}')

Output and plotting
===================

Phases that would otherwise draw to the screen can be made to behave in a
script or a server with the helpers in :mod:`pyCamSet.workflow.logs`:
``captured_output`` redirects what a phase prints into your ``log``, and
``non_interactive_plotting`` puts matplotlib on a non-interactive backend for
the duration.
