"""
Building a calibration target from a description of it.

A target is a class and the arguments it was constructed with.  That is not
a new idea here: :class:`~pyCamSet.calibration_targets.core.abstract_target.AbstractTarget`
records those arguments as ``input_args``, the detection pool rebuilds a
target from them in each worker process, and loading a camset rebuilds one
the same way.  This is that mechanism, named, so a phase can use it too.

A *spec* is those arguments plus the ``type`` that says which class they
belong to::

    {"type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
     "square_size": 4.0, "legacy": False}

Anything the class defaults, a spec may leave out.  Adding a target is one
line in :data:`TARGET_CLASSES` and nothing else.
"""
from __future__ import annotations

import importlib
from typing import Any

#: Every target a phase can build, by the name it is known by.  Held as
#: module and class names rather than as classes so that importing this
#: costs nothing and so that PuzzleBoard, whose detector is an optional
#: dependency, is only imported when one is actually asked for.
TARGET_CLASSES: dict[str, tuple[str, str]] = {
    "Ccube": ("pyCamSet.calibration_targets.ccube", "Ccube"),
    "Ccube2": ("pyCamSet.calibration_targets.ccube2", "Ccube2"),
    "CIco": ("pyCamSet.calibration_targets.cico", "CIco"),
    "CIco2": ("pyCamSet.calibration_targets.cico2", "CIco2"),
    "ChArUco": ("pyCamSet.calibration_targets.charuco", "ChArUco"),
    "ChArUco2": ("pyCamSet.calibration_targets.charuco2", "ChArUco2"),
    "PuzzleBoard": (
        "pyCamSet.calibration_targets.puzzleboard", "PuzzleBoard"),
    "PuzzleBoardCube": (
        "pyCamSet.calibration_targets.puzzleboard_cube", "PuzzleBoardCube"),
    "PuzzleBoardIco": (
        "pyCamSet.calibration_targets.puzzleboard_ico", "PuzzleBoardIco"),
}

#: The target names, in the order an interface should offer them.
TARGET_NAMES: tuple[str, ...] = tuple(TARGET_CLASSES)

#: What an interface calls a target, where that is not its registry name.
#: The names are persisted in every saved spec, so they stay as they are;
#: these only say which marker generation a board is printed with, which
#: the names on their own leave a reader to guess.
TARGET_LABELS: dict[str, str] = {
    "ChArUco": "ChArUco1",
    "Ccube": "ChArUco1 ccube",
    "ChArUco2": "ChArUco2",
    "Ccube2": "ChArUco2 ccube",
    "PuzzleBoardCube": "pcube",
    "CIco": "ChArUco1 icosahedron",
    "CIco2": "ChArUco2 icosahedron",
    "PuzzleBoardIco": "PuzzleBoard icosahedron",
}


def target_label(name: str) -> str:
    """
    What an interface shows for a target name.

    :param name: the target's name, as :data:`TARGET_CLASSES` keys it
    :return: its label, or the name itself for a target with no label
    """
    return TARGET_LABELS.get(name, name)


#: The key under which a spec names its class.
TYPE_KEY = "type"


def target_class(name: str) -> type:
    """
    Return the class a target name refers to.

    :param name: the target's name, as :data:`TARGET_CLASSES` keys it
    :raises ValueError: for a name no target answers to
    """
    try:
        module_name, class_name = TARGET_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"Unknown target type {name!r}; expected one of "
            f"{', '.join(TARGET_NAMES)}."
        ) from None
    return getattr(importlib.import_module(module_name), class_name)


def build_target(spec: dict[str, Any]):
    """
    Build the target a spec describes.

    :param spec: ``type`` plus the target's constructor arguments
    :raises ValueError: for an unknown type, or a spec that names none
    """
    if not spec or TYPE_KEY not in spec:
        raise ValueError(
            f"A target spec must say which target it is, under {TYPE_KEY!r}.")
    arguments = dict(spec)
    return target_class(arguments.pop(TYPE_KEY))(**arguments)


def spec_of(target) -> dict[str, Any]:
    """
    Return the spec that would rebuild *target*.

    The inverse of :func:`build_target`, for a target that already exists --
    a saved camset's, say, whose settings a phase wants to compare against.

    :param target: any :class:`AbstractTarget`
    :raises ValueError: for a target this build does not have a name for
    """
    name = type(target).__name__
    if name not in TARGET_CLASSES:
        raise ValueError(f"No target type is registered under {name!r}.")
    return {TYPE_KEY: name, **dict(target.input_args)}
