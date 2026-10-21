"""Check the wide conda search for Cairo registers only what cairocffi can load.
"""
from __future__ import annotations

import os
import sys

import pytest

from pyCamSet.utils import cairo_dll_helper as helper


class _Handle:
    def __init__(self, path):
        self.path = path
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def fake_conda(tmp_path, monkeypatch):
    """A conda root whose Library/bin has a recording stand-in for add_dll_directory."""
    root = tmp_path / "miniconda"
    lib_bin = root / "Library" / "bin"
    lib_bin.mkdir(parents=True)
    handles = []

    def add_dll_directory(path):
        handles.append(_Handle(path))
        return handles[-1]

    monkeypatch.setattr(os, "add_dll_directory", add_dll_directory, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(helper, "_library_bin_dirs", lambda: [lib_bin])
    return lib_bin, handles


def test_a_directory_with_only_other_cairo_dlls_is_not_registered(fake_conda, monkeypatch):
    lib_bin, handles = fake_conda
    (lib_bin / "cairo-gobject-2.dll").write_bytes(b"")
    (lib_bin / "cairo-script-interpreter-2.dll").write_bytes(b"")
    path_before = os.environ["PATH"]

    assert helper._cairosvg_from_conda_roots() is None
    assert handles == []
    assert os.environ["PATH"] == path_before


def test_a_failed_candidate_is_unregistered_and_path_is_restored(fake_conda, monkeypatch):
    lib_bin, handles = fake_conda
    (lib_bin / "cairo.dll").write_bytes(b"")
    path_before = os.environ["PATH"]
    monkeypatch.delitem(sys.modules, "cairosvg", raising=False)
    seen = {}

    import builtins
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "cairosvg":
            seen["path"] = os.environ["PATH"]
            raise OSError("no library called cairo was found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    assert helper._cairosvg_from_conda_roots() is None

    assert [h.path for h in handles] == [str(lib_bin)]
    assert handles[0].closed
    assert seen["path"].startswith(str(lib_bin))  # visible during the attempt only
    assert os.environ["PATH"] == path_before


def test_conda_roots_include_the_environment_and_never_a_drive_root(tmp_path, monkeypatch):
    env = tmp_path / "miniconda" / "envs" / "x"
    env.mkdir(parents=True)
    drive_root = tmp_path.anchor
    monkeypatch.setenv("CONDA_PREFIX", str(env))
    monkeypatch.setenv("CONDA_ROOT", drive_root)

    roots = helper._conda_roots()
    assert env in roots
    assert env.parent.parent in roots
    assert roots.index(env) < roots.index(env.parent.parent)
    assert not any(r.parent == r for r in roots)
