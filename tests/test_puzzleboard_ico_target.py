"""
What a PuzzleBoardIco is.

The target is a window of PuzzleBoard's periodic code clipped to a triangle,
twenty times.  Two things decide whether that works: the windows must not
overlap, because two faces sharing any of the field print identical code; and
what is printed must decode back to the window it was cut from, which is where
the half-square difference between this module's lattice and upstream's lives.
Both are checked here.
"""
import re

import numpy as np
import pytest

from pyCamSet import calibration_targets
from pyCamSet.calibration_targets import TARGET_NAMES
from pyCamSet.calibration_targets.core.abstract_target import EXPORT_KINDS
from pyCamSet.calibration_targets.core.parameters import DocumentedParameters
from pyCamSet.calibration_targets.core.target_registry import TARGET_LABELS
from pyCamSet.calibration_targets.puzzleboard import _CODE_SIZE
from pyCamSet.calibration_targets.puzzleboard_ico import (
    MAX_FACE_SQUARES,
    PuzzleBoardIco,
)


def _skip_without_cairo():
    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError) as error:
        pytest.skip(f"native cairo is unavailable: {error}")


def _skip_without_detector():
    try:
        import puzzle_board  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("puzzle_board is not installed")


@pytest.fixture(scope="module")
def target():
    return PuzzleBoardIco(length=100.0, n_points=16)


def _render(target, face_index, width=1600):
    import cv2
    return cv2.cvtColor(
        target._face_texture(face_index, width), cv2.COLOR_RGB2GRAY)


# -- what it is ---------------------------------------------------------------

def test_puzzleboard_ico_is_registered_labelled_and_exported():
    assert "PuzzleBoardIco" in list(TARGET_NAMES)
    assert TARGET_LABELS["PuzzleBoardIco"] == "PuzzleBoard icosahedron"
    assert "PuzzleBoardIco" in calibration_targets.__all__
    assert calibration_targets.PuzzleBoardIco is PuzzleBoardIco


def test_it_declares_the_arguments_that_decide_its_geometry():
    offered = PuzzleBoardIco.construction_parameters()
    assert isinstance(offered, DocumentedParameters)
    assert [p.key for p in offered.parameters] == ["n_points", "length"]


def test_a_printable_is_named_for_what_it_is():
    assert PuzzleBoardIco.printable_name(
        {"n_points": 16, "length": 100.0}, "svg"
    ) == "puzzleboard_ico_16points_100mm.svg"


# -- the windows --------------------------------------------------------------

def test_no_two_faces_share_any_of_the_code_field(target):
    """
    The one placement mistake that costs anything.

    Two windows that overlapped would print identical code on two faces, and a
    patch in the shared part would decode to both.  Spacing beyond disjoint
    buys nothing -- the base code is unique but has no margin -- so this is the
    property the layout is actually for.
    """
    size = target.n_points
    origins = target.face_origins
    assert len(origins) == 20
    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            (ax, ay), (bx, by) = origins[i], origins[j]
            apart = abs(ax - bx) >= size or abs(ay - by) >= size
            assert apart, f"windows {i} and {j} overlap"


def test_every_window_lies_inside_the_code_field(target):
    for x, y in target.face_origins:
        assert 0 <= x and x + target.n_points <= _CODE_SIZE
        assert 0 <= y and y + target.n_points <= _CODE_SIZE


def test_a_face_too_large_for_twenty_windows_is_refused():
    with pytest.raises(ValueError, match="must not exceed"):
        PuzzleBoardIco(length=100.0, n_points=MAX_FACE_SQUARES + 1)


def test_a_face_too_coarse_to_hold_a_corner_is_refused():
    with pytest.raises(ValueError, match="at least"):
        PuzzleBoardIco(length=100.0, n_points=4)


def test_an_icosahedron_with_no_size_is_refused():
    with pytest.raises(ValueError, match="edge length"):
        PuzzleBoardIco(length=0.0, n_points=16)


# -- the geometry -------------------------------------------------------------

def test_every_face_carries_the_same_lattice(target):
    assert target.point_data.shape == (20, target.points_per_face, 3)
    assert target.point_local.shape == target.point_data.shape


def test_each_face_is_flat(target):
    for face in target.point_data:
        centred = face - face.mean(axis=0)
        assert np.linalg.svd(centred, compute_uv=False)[-1] < 1e-9


def test_the_points_sit_inside_the_face_they_belong_to(target):
    corners = target.basis.face_corners(target.length)
    for face_index, face in enumerate(target.point_data):
        triangle = corners[face_index]
        origin = triangle[0]
        basis = np.stack([triangle[1] - origin, triangle[2] - origin], axis=1)
        coefficients, *_ = np.linalg.lstsq(basis, (face - origin).T, rcond=None)
        a, b = coefficients
        assert np.all(a > -1e-9) and np.all(b > -1e-9)
        assert np.all(a + b < 1 + 1e-9)


# -- reading one back ---------------------------------------------------------

def test_a_printed_window_decodes_to_the_window_it_was_cut_from(target):
    """
    This is what ``_CODE_PHASE`` is for.

    Upstream centres its squares on the integer code positions and this module
    runs them integer to integer, which shifts the printed pattern by half a
    square -- a whole number of code positions, absorbed when the code is read.
    Get it wrong and every face decodes as a different one, so the check is
    that a face reads back as itself.
    """
    _skip_without_cairo()
    _skip_without_detector()

    for face_index in (0, 7, 13, 19):
        detection = target.find_in_image(_render(target, face_index))
        assert detection.has_data, f"face {face_index} read as nothing"
        keys = np.asarray(detection.keys)
        assert set(keys[:, 0].tolist()) == {face_index}
        assert len(keys) == target.points_per_face


def test_a_printed_face_puts_its_corners_where_it_says_they_are(target):
    _skip_without_cairo()
    _skip_without_detector()

    width = 1600
    detection = target.find_in_image(_render(target, 0, width))
    scale = width / target.length
    expected = (target.live_corners * target.square_size) * scale
    keys = np.asarray(detection.keys)
    found = np.asarray(detection.image_points)
    error = np.linalg.norm(found - expected[keys[:, 1]], axis=1)
    assert error.max() < 0.05 * target.square_size * scale


def test_the_printed_net_gives_back_every_face(target):
    """
    The net is what is printed, so reading it back checks the whole chain at
    once: the clip, the code phase, the window layout and the net.
    """
    _skip_without_cairo()
    _skip_without_detector()
    from io import BytesIO

    import cairosvg
    from PIL import Image

    drawing, _, _ = target._svg_document(
        border_width=5.0, draw_cut_outline=False, draw_face_ids=False)
    png = cairosvg.svg2png(
        bytestring=drawing.tostring().encode("utf-8"), output_width=6000)
    with Image.open(BytesIO(png)) as image:
        net = np.asarray(image.convert("L"))

    keys = np.asarray(target.find_in_image(net).keys)
    assert sorted(set(keys[:, 0].tolist())) == list(range(20))
    for face_index in range(20):
        assert int(np.sum(keys[:, 0] == face_index)) == target.points_per_face


# -- printing -----------------------------------------------------------------

@pytest.mark.parametrize("kind", EXPORT_KINDS)
def test_it_writes_itself_as_every_format_it_offers(target, kind, tmp_path):
    _skip_without_cairo()
    written = target.save_printable(tmp_path / f"net_{kind}", kind=kind)
    assert written.exists() and written.stat().st_size > 1024
    if kind.startswith("pdf"):
        embedded = bool(re.search(rb"/Subtype\s*/Image", written.read_bytes()))
        assert embedded == (kind == "pdf_raster")


def test_a_format_it_is_not_is_refused(target, tmp_path):
    with pytest.raises(ValueError, match="cannot be written as"):
        target.save_printable(tmp_path / "net", kind="dxf")


def test_the_solid_writes_itself_for_printing(target, tmp_path):
    pv = pytest.importorskip("pyvista")
    mesh = pv.read(str(target.to_stl(tmp_path / "core")))
    assert mesh.n_points == 12
    assert mesh.n_faces == 20


# -- the spec round trip ------------------------------------------------------

def test_it_can_be_rebuilt_from_what_it_was_made_with(target):
    from pyCamSet.calibration_targets.core.target_registry import (
        build_target, spec_of)

    spec = spec_of(target)
    assert spec["type"] == "PuzzleBoardIco"
    assert np.allclose(build_target(spec).point_data, target.point_data)
