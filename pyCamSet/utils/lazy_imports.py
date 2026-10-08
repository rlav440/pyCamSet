"""
Names a module used to import eagerly, resolved on first access instead.

Every calibration target is built from :mod:`pyCamSet.calibration_targets.core`,
and that used to import matplotlib, scipy, pyvista and numba at module scope
on the way -- well over a second and a hundred megabytes for a caller that
only wants to find a board in an image, which needs none of them.  The
modules on that path now import those where they are used.

What that would quietly break is a caller that reached into one of those
modules for a name it had imported, ``general_utils.plt`` say.  A module
keeps those names by setting its ``__getattr__`` (PEP 562) to
:func:`lazy_attributes`, which imports them the first time one is asked for.
Inside the module itself nothing changes: a function that needs one imports
it, since a module's own global lookups never reach its ``__getattr__``.

A name whose package is missing behaves as the module's own import made it
behave.  Where that import was guarded -- ``pv = None`` when pyvista is not
installed -- the name is marked optional and is ``None`` then.  Every other
name raises the ImportError on access; before, importing the module did.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Collection


def lazy_attributes(
        module_name: str,
        attributes: dict[str, tuple[str, str | None]],
        optional: Collection[str] = (),
) -> Callable[[str], Any]:
    """
    Make a module ``__getattr__`` that imports the given names on demand.

    :param module_name: the module's ``__name__``, for the error message
    :param attributes: each name, mapped to the module it comes from and the
        attribute of that module it is -- or ``None`` for the module itself
    :param optional: the names that are ``None`` when their module cannot be
        imported, rather than raising its ImportError
    :return: the function to bind as the module's ``__getattr__``
    """
    def __getattr__(name: str) -> Any:
        try:
            source, attribute = attributes[name]
        except KeyError:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}") from None
        try:
            value = importlib.import_module(source)
        except ImportError:
            if name in optional:
                return None
            raise
        return value if attribute is None else getattr(value, attribute)

    return __getattr__
