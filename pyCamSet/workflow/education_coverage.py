"""Record how education mode teaches each registered target, lens model, detector backend, solver and loss.

A proof is ``tests/<file>.py::<test_name>``, optionally with a parametrisation id in brackets
(``tests/<file>.py::<test_name>[<id>]``). The contract test parses the file with ``ast``, so the
test must exist. For a ``taught`` entry every proof must also name the item (its registry name,
or its one-word label) in the test's source, or in the bracketed id.
"""
from __future__ import annotations

from dataclasses import dataclass

#: A named teaching path exists in the education code and a test that names the item exercises it.
TAUGHT = "taught"
#: Taught only through generic wording, or by a path no test exercises for this item. Reason given.
PARTIAL = "partial"
#: No teaching path. Reason given.
NOT_YET = "not_yet_taught"


@dataclass(frozen=True)
class Coverage:
    """How one registered item is taught.

    :param status: ``TAUGHT``, ``PARTIAL`` or ``NOT_YET``.
    :param where: ``package.module:name`` (or ``package.module:Class.method``) of the education code
        that teaches the item. Each must import and resolve.
    :param proof: ``tests/<file>.py::<test_name>`` references (see the module docstring).
    :param reason: why the status is not ``TAUGHT`` (required for those statuses).
    """

    status: str
    where: tuple[str, ...] = ()
    proof: tuple[str, ...] = ()
    reason: str = ""


_DETECT = "pyCamSet.workflow.education_detect_explain"
_PHASE2 = "pyCamSet.workflow.education_visual_p2"
_PHASE3 = "pyCamSet.workflow.education_visual_p3"
_PHASE4 = "pyCamSet.workflow.education_visual_p4"
_DETECT_TEST = "tests/test_education_detect_explain.py"
#: The detector tests are parametrised over the synthetic cases; the case id names the target.
_DETECT_CASES = f"{_DETECT_TEST}::test_explanation_steps_and_consistency"

#: Keys per kind are the registry names: TARGET_CLASSES keys, MODEL_BLOCKS class names,
#: DETECTOR_BACKENDS keys, SOLVER_CHOICES values and loss names (ROBUST_LOSSES plus "linear").
COVERAGE: dict[str, dict[str, Coverage]] = {
    "target": {
        "ChArUco": Coverage(TAUGHT, (f"{_DETECT}:_explain_charuco",),
                            (f"{_DETECT_CASES}[charuco_aruco1]",)),
        "ChArUco2": Coverage(TAUGHT, (f"{_DETECT}:_explain_grid_board",),
                             (f"{_DETECT_CASES}[charuco2]",)),
        "Ccube": Coverage(TAUGHT, (f"{_DETECT}:_explain_charuco",),
                          (f"{_DETECT_CASES}[ccube_aruco1]",)),
        "Ccube2": Coverage(TAUGHT, (f"{_DETECT}:_explain_grid_board",),
                           (f"{_DETECT_CASES}[ccube2]",)),
        "PuzzleBoard": Coverage(
            PARTIAL, (f"{_DETECT}:_explain_puzzleboard",),
            ("tests/test_education_phase1.py::test_final_overlay_scale_follows_the_effective_rescale_setting[PuzzleBoard]",),
            reason="The explanation path is the one PuzzleBoardCube uses, but no test runs it on a "
                   "flat PuzzleBoard. The PuzzleBoard case in test_education_phase1.py checks only "
                   "the overlay's rescale scale, not the explanation. The real-run recompute test "
                   "uses the PuzzleBoardCube (pcube) run."),
        "PuzzleBoardCube": Coverage(
            TAUGHT, (f"{_DETECT}:_explain_puzzleboard",),
            (f"{_DETECT_TEST}::test_rescale_and_gamma_geometry_maps_to_native_pixels",)),
        "PuzzleBoardIco": Coverage(
            NOT_YET,
            reason="puzzleboard_ico.py emits no detector explain events, so explain_detection "
                   "reports backend 'unknown' with only the input and result steps. The release "
                   "matrix runs its cells but nothing explains them."),
        "CIco": Coverage(
            NOT_YET,
            reason="cico.py emits no detector explain events, so explain_detection reports "
                   "backend 'unknown' with only the input and result steps."),
        "CIco2": Coverage(
            NOT_YET,
            reason="cico2.py emits no detector explain events, so explain_detection reports "
                   "backend 'unknown' with only the input and result steps."),
    },
    "lens_model": {
        "Camera": Coverage(
            PARTIAL, (f"{_PHASE2}:lens_summary", f"{_PHASE2}:face_warp"),
            ("tests/test_education_phase2_visual.py::test_lens_numbers_and_distortion_grid_are_from_the_saved_lens",
             "tests/test_education_phase2_visual.py::test_pinhole_views_are_posed_in_closed_form_or_by_the_older_pose_only_refit"),
            reason="The pinhole lens summary and distortion grid are tested on a SimpleNamespace "
                   "stand-in, not on a Camera object. The pinhole Phase 2 check is covered only by "
                   "noiseless synthetic views. The one face-warp test uses the telecentric fit. No "
                   "education test names Camera."),
        "TelecentricCamera": Coverage(
            TAUGHT,
            (f"{_PHASE2}:is_telecentric", f"{_PHASE2}:_telecentric_fit",
             f"{_PHASE2}:display_depths", f"{_PHASE3}:TELECENTRIC_DISPLAY_DISTANCE",
             "pyCamSet.workflow.education_derive:_phase4_point_classes"),
            ("tests/test_education_derive_p4.py::test_telecentric_translation_does_not_invent_viewing_ray_diversity",)),
    },
    "detector_backend": {
        "aruco1": Coverage(TAUGHT, (f"{_DETECT}:_explain_charuco",),
                           (f"{_DETECT_CASES}[charuco_aruco1]",)),
        "aruco2": Coverage(
            TAUGHT, (f"{_DETECT}:_explain_charuco", f"{_DETECT}:_explain_grid_board"),
            (f"{_DETECT_CASES}[charuco_aruco2]",)),
        "puzzle_board": Coverage(
            TAUGHT, (f"{_DETECT}:_explain_puzzleboard",),
            (f"{_DETECT_TEST}::test_rescale_and_gamma_geometry_maps_to_native_pixels",)),
    },
    "solver": {
        "schur": Coverage(
            TAUGHT,
            (f"{_PHASE3}:schur_structure", f"{_PHASE3}:reduced_camera_system",
             "pyCamSet.workflow.education_playback:AcceptedStepCollector.add_schur"),
            ("tests/test_education_phase3_visual.py::test_reduced_camera_system_is_the_schur_complement_of_the_jacobian",)),
        "trf": Coverage(
            PARTIAL, ("pyCamSet.workflow.education_playback:read_accepted_step_trace",),
            ("tests/test_solver_telemetry.py::test_endpoints_only_fallback_is_not_playable",),
            reason="The 'steps not recorded for this solver' branch of read_accepted_step_trace is "
                   "tested, with a trf payload, in test_solver_telemetry.py. Not covered: the Phase 3 "
                   "caption that says which solver a run used (it is added when the solver is not "
                   "schur), which no test asserts."),
    },
    "loss": {
        "linear": Coverage(
            TAUGHT,
            ("pyCamSet.workflow.education_trace:loss_settings_from_options",
             "pyCamSet.gui.education_phase4:cost_column_title"),
            ("tests/test_education_trace.py::test_loss_settings_preserve_f_scale_and_default_legacy_loss",)),
        "soft_l1": Coverage(
            TAUGHT,
            ("pyCamSet.gui.education_phase3:_objective_header",
             "pyCamSet.gui.education_phase4:cost_column_title",
             "pyCamSet.workflow.education_playback:_cost_from_residuals"),
            ("tests/test_education_phase4_visual.py::test_cost_column_names_the_loss_it_holds",)),
        "huber": Coverage(
            TAUGHT,
            ("pyCamSet.gui.education_phase3:_objective_header",
             "pyCamSet.gui.education_phase4:cost_column_title",
             "pyCamSet.workflow.education_playback:_cost_from_residuals"),
            ("tests/test_education_phase4.py::test_phase4_provenance_and_loss_are_read_from_state_source",)),
        "cauchy": Coverage(
            PARTIAL,
            ("pyCamSet.gui.education_phase3:_objective_header",
             "pyCamSet.gui.education_phase4:cost_column_title"),
            ("tests/test_education_phase4_visual.py::test_cost_column_names_the_loss_it_holds",),
            reason="Taught only by the generic non-linear wording (any loss other than linear is "
                   "named). The test covers soft_l1, not cauchy. No education test names cauchy."),
        "arctan": Coverage(
            PARTIAL,
            ("pyCamSet.gui.education_phase3:_objective_header",
             "pyCamSet.gui.education_phase4:cost_column_title"),
            ("tests/test_education_phase4_visual.py::test_cost_column_names_the_loss_it_holds",),
            reason="Taught only by the generic non-linear wording (any loss other than linear is "
                   "named). The test covers soft_l1, not arctan. No education test names arctan."),
    },
}
