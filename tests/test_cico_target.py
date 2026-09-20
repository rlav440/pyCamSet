"""
What a CIco is.

The target is a square ChArUco board clipped to a triangle, twenty times, so
the checks that matter are the ones about clipping: that the points a face
claims are the corners it actually printed, that a face decodes as itself and
not as one of the other nineteen, and that the marker alphabet is shared out
twenty ways without running past its end.
"""
import re

import numpy as np
import pytest

import pyCamSet
from pyCamSet import calibration_targets
from pyCamSet.calibration_targets import TARGET_NAMES
from pyCamSet.calibration_targets.cico import CIco
from pyCamSet.calibration_targets.core.abstract_target import EXPORT_KINDS
from pyCamSet.calibration_targets.core.parameters import DocumentedParameters
from pyCamSet.calibration_targets.core.target_registry import TARGET_LABELS


def _skip_without_cairo():
    """cairosvg raises OSError, not ImportError, when its library is absent."""
    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError) as error:
        pytest.skip(f"native cairo is unavailable: {error}")


@pytest.fixture(scope="module")
def target():
    return CIco(length=100.0, n_points=10)


# -- what it is ---------------------------------------------------------------

def test_cico_is_registered_labelled_and_exported():
    names = list(TARGET_NAMES)
    assert "CIco" in names
    assert TARGET_LABELS["CIco"] == "ChArUco1 icosahedron"
    assert "CIco" in calibration_targets.__all__
    assert calibration_targets.CIco is CIco


def test_cico_offers_the_detectors_a_charuco_board_can_be_read_with():
    assert list(CIco.DETECTOR_BACKENDS) == ["aruco1", "aruco2"]


def test_cico_declares_the_arguments_that_decide_its_geometry():
    offered = CIco.construction_parameters()
    assert isinstance(offered, DocumentedParameters)
    assert [p.key for p in offered.parameters] == [
        "n_points", "length", "border_fraction", "aruco_dict", "legacy"]


def test_a_printable_is_named_for_what_it_is():
    assert CIco.printable_name(
        {"n_points": 10, "length": 100.0}, "pdf_raster") == "cico_10points_100mm.pdf"


# -- the geometry -------------------------------------------------------------

def test_every_face_carries_the_same_lattice(target):
    assert target.point_data.shape == (20, target.points_per_face, 3)
    assert target.points_per_face > 0
    # make_local reshapes to (-1, n, 3), so an uneven face count is not a
    # thing this target is allowed to have.
    assert target.point_local.shape == target.point_data.shape


def test_each_face_is_flat(target):
    for face in target.point_data:
        centred = face - face.mean(axis=0)
        # The smallest singular value is the out-of-plane spread.
        assert np.linalg.svd(centred, compute_uv=False)[-1] < 1e-9


def test_the_points_sit_inside_the_face_they_belong_to(target):
    """
    Clipping is the whole idea, so a point outside its triangle is the bug
    this target exists to avoid.
    """
    corners = target.basis.face_corners(target.length)
    for face_index, face in enumerate(target.point_data):
        triangle = corners[face_index]
        # Barycentric coordinates against the face's own triangle.
        origin = triangle[0]
        u, v = triangle[1] - origin, triangle[2] - origin
        basis = np.stack([u, v], axis=1)
        coefficients, *_ = np.linalg.lstsq(basis, (face - origin).T, rcond=None)
        a, b = coefficients
        assert np.all(a > -1e-9) and np.all(b > -1e-9)
        assert np.all(a + b < 1 + 1e-9)


def test_a_clipped_face_keeps_only_corners_it_printed(target):
    """
    Every point the target claims is a corner of four printed squares.
    """
    printed = {(int(c), int(r)) for c, r in target.cells}
    for column, row in target.live_corners:
        for dx in (-1, 0):
            for dy in (-1, 0):
                assert (column + dx, row + dy) in printed


# -- the marker alphabet ------------------------------------------------------

def test_the_dictionary_is_shared_out_twenty_ways(target):
    """
    Each face gets its own slice of the alphabet, as a Ccube's does.

    The slice is a dictionary of its own, so a board's ids stay local -- face
    k's marker j is the parent dictionary's ``k * markers_per_face + j``, and
    it is the bytes that differ between faces, not the numbering.  The aruco2
    path is the one that sees the global id, and divides it back out.
    """
    from pyCamSet.calibration_targets.markers.aruco2 import resolve_dictionary

    assert len(target.a_dicts) >= 20
    assert len(target.boards) == 20

    parent = resolve_dictionary(target.aruco_dict, target.marker_backend)
    per_face = target.markers_per_face
    for face_index in (0, 1, 19):
        expected = parent.bytesList[
            face_index * per_face:(face_index + 1) * per_face]
        assert np.array_equal(target.a_dicts[face_index].bytesList, expected)

    # No face shares a marker with any other.
    assert not np.array_equal(
        target.a_dicts[0].bytesList, target.a_dicts[1].bytesList)


def test_a_face_too_fine_for_the_dictionary_is_refused():
    with pytest.raises(ValueError, match="markers, twenty faces"):
        CIco(length=100.0, n_points=20)


def test_a_face_too_coarse_to_hold_a_corner_is_refused():
    with pytest.raises(ValueError, match="at least"):
        CIco(length=100.0, n_points=4)


@pytest.mark.parametrize("values, refused", [
    ({"length": 0.0}, "edge length"),
    ({"border_fraction": 0.0}, "neither all of it nor none"),
    ({"border_fraction": 1.0}, "neither all of it nor none"),
])
def test_an_impossible_icosahedron_is_refused(values, refused):
    with pytest.raises(ValueError, match=refused):
        CIco(**{"length": 100.0, "n_points": 10, **values})


# -- reading one back ---------------------------------------------------------

def test_a_printed_face_decodes_as_that_face_and_no_other(target):
    """
    Twenty faces are cut from one alphabet, so the thing that can go wrong is
    a face decoding as a different one.  Each face is rendered as it prints
    and read back on its own.
    """
    _skip_without_cairo()
    import cv2

    for face_index in (0, 7, 13, 19):
        texture = target._face_texture(face_index, 1400)
        grey = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)
        detection = target.find_in_image(grey)
        assert detection.has_data, f"face {face_index} read as nothing"
        keys = np.asarray(detection.keys)
        assert set(keys[:, 0].tolist()) == {face_index}
        assert len(keys) == target.points_per_face


def test_a_printed_face_puts_its_corners_where_it_says_they_are(target):
    """
    The points a face reports and the points it prints are the same points.
    """
    _skip_without_cairo()
    import cv2

    resolution = 1400
    texture = target._face_texture(0, resolution)
    grey = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)
    detection = target.find_in_image(grey)

    scale = resolution / target.length
    expected = ((target.live_corners * target.square_size)
                + target.board_offset) * scale
    keys = np.asarray(detection.keys)
    found = np.asarray(detection.image_points)
    error = np.linalg.norm(found - expected[keys[:, 1]], axis=1)
    # A square is square_size * scale pixels across; the corners land well
    # inside a hundredth of one.
    assert error.max() < 0.02 * target.square_size * scale


def test_the_printed_net_gives_back_every_face(target):
    """
    The net is what is actually printed, so reading it back is the check that
    the whole chain agrees: the lattice, the clip, the dictionary split and
    the layout.
    """
    _skip_without_cairo()
    from io import BytesIO

    import cairosvg
    from PIL import Image

    drawing, _, _ = target._svg_document(
        border_width=5.0, draw_cut_outline=False, draw_face_ids=False)
    png = cairosvg.svg2png(
        bytestring=drawing.tostring().encode("utf-8"), output_width=4000)
    with Image.open(BytesIO(png)) as image:
        net = np.asarray(image.convert("L"))

    detection = target.find_in_image(net)
    keys = np.asarray(detection.keys)
    assert sorted(set(keys[:, 0].tolist())) == list(range(20))
    for face_index in range(20):
        assert int(np.sum(keys[:, 0] == face_index)) == target.points_per_face


# -- printing -----------------------------------------------------------------

@pytest.mark.parametrize("kind", EXPORT_KINDS)
def test_an_icosahedron_writes_itself_as_every_format_it_offers(
        target, kind, tmp_path):
    _skip_without_cairo()
    written = target.save_printable(tmp_path / f"net_{kind}", kind=kind)
    assert written.exists()
    assert written.stat().st_size > 1024

    if kind.startswith("pdf"):
        embedded = bool(re.search(rb"/Subtype\s*/Image", written.read_bytes()))
        assert embedded == (kind == "pdf_raster"), (
            "a vector PDF must not embed a raster, and a raster one must")


def test_a_format_an_icosahedron_is_not_is_refused(target, tmp_path):
    with pytest.raises(ValueError, match="cannot be written as"):
        target.save_printable(tmp_path / "net", kind="dxf")


def test_the_solid_writes_itself_for_printing(target, tmp_path):
    pv = pytest.importorskip("pyvista")
    written = target.to_stl(tmp_path / "core")
    mesh = pv.read(str(written))
    assert mesh.n_points == 12
    assert mesh.n_faces == 20

    # The printed core is the solid the target's own points sit on.
    written_centres = np.sort(mesh.cell_centers().points, axis=0)
    expected = np.sort(
        target.basis.face_corners(target.length * 1000.0).mean(axis=1), axis=0)
    assert np.allclose(written_centres, expected, atol=1e-4)


# -- the spec round trip ------------------------------------------------------

def test_an_icosahedron_can_be_rebuilt_from_what_it_was_made_with(target):
    """Workers rebuild a target from its spec, so the spec has to be enough."""
    from pyCamSet.calibration_targets.core.target_registry import (
        build_target, spec_of)

    spec = spec_of(target)
    assert spec["type"] == "CIco"
    rebuilt = build_target(spec)
    assert np.allclose(rebuilt.point_data, target.point_data)
