"""
The calibration workflow: the five phases, and the workspace they live in.

Nothing here imports a user interface toolkit.  A phase is a function that
takes settings, a workspace and somewhere to send its output, and returns the
run record it saved::

    from pyCamSet.workflow import WorkspaceManager, phase1

    workspace = WorkspaceManager(workspace_path_for("/data/rig"))
    run = phase1.run(params, workspace, log=print)

The GUI is one caller of these; a script is another.
"""
from pyCamSet.workflow import phase1, phase2, phase3, phase4
from pyCamSet.workflow.logs import captured_output, discard, non_interactive_plotting
from pyCamSet.workflow.params import ParamError
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    make_run_id,
    workspace_path_for,
)

__all__ = [
    "ParamError",
    "WorkspaceManager",
    "captured_output",
    "discard",
    "make_run_id",
    "non_interactive_plotting",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "workspace_path_for",
]
