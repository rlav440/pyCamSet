"""Ccube2 (a cube of ChArUco2 grid-board faces) target regressions.

Gates itself on the optional ``aruco2`` package, as
``test_charuco2_target.py`` does: a Ccube2 cannot be built at all without
it, so the module skips itself before its other imports.

Each face is printed from the same vector layout as a ChArUco2 board
(``charuco2/layout.py``), so a face texture is checked against aruco2's own
``get_grid_board_image`` pixel for pixel at a scale where every cell and band
edge lands on a whole pixel (see that module's docstring for which scales
those are). The cube itself is Ccube's, so the end-to-end check here is
geometric: a synthetic camera photographs the cube's textures placed where
``point_data`` says the faces are, and every detected corner must land where
its ``point_data`` projects.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

aruco2 = pytest.importorskip("aruco2")

from pyCamSet.calibration_targets.ccube2 import target as ccube2_target_module
from pyCamSet.calibration_targets.ccube2.target import Ccube2
from pyCamSet.calibration_targets.ccube.target import NET_FORMS
from pyCamSet.calibration_targets.charuco2 import layout
from pyCamSet.calibration_targets.charuco2.target import ChArUco2
from pyCamSet.calibration_targets.core.abstract_target import EXPORT_KINDS
from pyCamSet.calibration_targets.core.target_registry import (
    TARGET_LABELS, TARGET_NAMES,
)
from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
    dictionary_marker_bits,
    render_grid_board_image,
)


def _skip_without_cairo():
    """Skip when the native cairo library is missing, as it is on test CI.

    cairosvg raises OSError, not ImportError, when the library it binds to is
    absent, so importorskip does not catch it.
    """
    try:
        import pyCamSet.utils.cairo_dll_helper  # noqa: F401
        import cairosvg  # noqa: F401
    except (ImportError, OSError) as err:
        pytest.skip(f"native cairo is unavailable: {err}")


def _lattice(cube: Ccube2) -> np.ndarray:
    """Every face's corners, row-major, in face-local metres."""
    return layout.grid_board_corners(
        cube.grid_size, cube.square_size, origin=(cube.margin, cube.margin))


def _texture_pixels(cube: Ccube2, local_xy: np.ndarray) -> np.ndarray:
    """Face-local metres to texture pixel coordinates, in the rasteriser's
    own pixel-*centre* convention (``layout.rasterise_rectangles``: pixel
    j's centre sits at ``(j + 0.5) / px_per_unit``), so a point on the
    boundary between pixels n-1 and n is at n - 0.5. This describes how the
    texture is drawn, not what a detector reports -- ``detect_grid_board_
    corners`` reports the pixel-*corner* convention instead (its docstring),
    0.5 px on from this. Used unmodified to build homography control points
    for warping a texture (a rasterisation fact); callers comparing this
    against a detection add the 0.5 px back themselves.
    """
    return local_xy * (cube.draw_res[0] / cube.length) - 0.5


def _svg_size_mm(text: str) -> tuple[float, float]:
    w_mm = float(re.search(r'width="([\d.]+)mm"', text).group(1))
    h_mm = float(re.search(r'height="([\d.]+)mm"', text).group(1))
    return w_mm, h_mm


def _rasterise_svg(svg_path: Path, px_per_mm: float) -> np.ndarray:
    import cairosvg
    from PIL import Image

    w_mm, h_mm = _svg_size_mm(svg_path.read_text(encoding="utf-8"))
    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=int(round(w_mm * px_per_mm)),
        output_height=int(round(h_mm * px_per_mm)),
        background_color="white")
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))


# -- what it is ---------------------------------------------------------------


def test_ccube2_is_registered_labelled_and_exported() -> None:
    import pyCamSet
    from pyCamSet import calibration_targets

    names = list(TARGET_NAMES)
    assert names.index("Ccube2") == names.index("Ccube") + 1
    assert TARGET_LABELS["Ccube2"] == "ChArUco2 ccube"
    assert "Ccube2" in pyCamSet.__all__
    assert "Ccube2" in calibration_targets.__all__
    assert pyCamSet.Ccube2 is Ccube2
    assert calibration_targets.Ccube2 is Ccube2


def test_ccube2_is_read_with_aruco2_only() -> None:
    from pyCamSet.calibration.camera_calibrator import detector_backend_of
    from pyCamSet.workflow.detections import detection_cache_name
    from pyCamSet.workflow.targets import detector_backend_of_spec

    assert list(Ccube2.DETECTOR_BACKENDS) == ["aruco2"]
    assert Ccube2.DETECTOR_BACKENDS["aruco2"].parameters == ()
    assert detector_backend_of_spec({"type": "Ccube2"}) == "aruco2"
    cube = Ccube2(n_points=3, length=20.0, border_fraction=0.2)
    assert detector_backend_of(cube) == "aruco2"
    assert detection_cache_name(1, detector_backend_of(cube)) == \
        "detected_datapoints_aruco2.pickle"


def test_ccube2_offers_what_charuco2_offers() -> None:
    offered = Ccube2.construction_parameters()
    assert [p.key for p in offered.parameters] == [
        "n_points", "length", "border_fraction", "aruco_dict"]
    assert offered.parameter("aruco_dict").choice_labels() == \
        ChArUco2.construction_parameters().parameter("a_dict").choice_labels()
    assert Ccube2.printable_name({"n_points": 6, "length": 40.0}, "pdf_raster") == \
        "ccube2_6points_40mm.pdf"


def test_ccube2_builds_a_cube_of_row_major_face_lattices() -> None:
    cube = Ccube2(length=40.0, n_points=5)
    n = 5
    assert cube.point_data.shape == (6, (n + 1) ** 2, 3)

    half = cube.length / 2
    # Ccube's TFORMS give their rotation vectors to 8 decimals, a rotation
    # error of up to ~5e-9 rad: across a 40 mm face that moves a corner off
    # its plane by ~3e-10 m (measured: 2.8e-10 at worst). 1e-7 of the edge
    # (4 nm) allows for that and still refuses any real misplacement.
    tol = 1e-7 * cube.length
    for face in cube.point_data:
        # Every face lies in one plane of the cube's surface...
        axis = int(np.argmin(np.ptp(face, axis=0)))
        assert np.ptp(face[:, axis]) < tol
        assert abs(abs(face[0, axis]) - half) < tol
        # ...with its corners a square_size apart along rows and columns.
        grid = face.reshape(n + 1, n + 1, 3)
        assert np.allclose(np.linalg.norm(np.diff(grid, axis=1), axis=-1), cube.square_size)
        assert np.allclose(np.linalg.norm(np.diff(grid, axis=0), axis=-1), cube.square_size)
        # The board sits in from the face's edge by the margin.
        assert np.allclose(np.abs(face).max(axis=0)[np.arange(3) != axis],
                           half - cube.margin)
    assert len({tuple(np.round(f.mean(axis=0), 9)) for f in cube.point_data}) == 6


@pytest.mark.parametrize("n_points", [3, 4, 6])
def test_ccube2_folds_its_faces_as_a_ccube_does(n_points) -> None:
    """A face's lattice sits on the cube exactly as a Ccube's does: its
    interior corners are a Ccube's corners, in the same order. A lattice
    turned or mirrored within its face keeps every spacing and plane, so
    only this pins which physical corner a gid is."""
    from pyCamSet.calibration_targets.ccube.target import Ccube

    cube = Ccube2(n_points=n_points, length=40.0, border_fraction=0.2,
                  draw_res=(200, 200))
    ccube = Ccube(n_points=n_points, length=40.0, border_fraction=0.2)
    n = n_points
    interior = cube.point_data.reshape(6, n + 1, n + 1, 3)[:, 1:n, 1:n]
    assert np.allclose(interior.reshape(6, -1, 3), ccube.point_data,
                       rtol=0, atol=1e-7 * cube.length)


def test_ccube2_faces_have_disjoint_marker_ids() -> None:
    cube = Ccube2(n_points=4, length=20.0, border_fraction=0.15)
    assert cube.face_ids == [list(range(k * 16, (k + 1) * 16)) for k in range(6)]


@pytest.mark.parametrize("values,refused", [
    ({"n_points": 1}, "at least 2x2"),
    ({"length": 0.0}, "has an edge length"),
    ({"border_fraction": 0.0}, "neither all of it nor none"),
    ({"border_fraction": 1.0}, "neither all of it nor none"),
    ({"n_points": 4, "border_fraction": 0.1}, r"at least 1/9 = 0\.1112"),
    ({"n_points": 8, "border_fraction": 0.05}, r"at least 1/17 = 0\.0589"),
    # Passes the 1/12 a 5.5 would need, not the 1/11 the 5x5 built needs.
    ({"n_points": 5.5, "border_fraction": 0.085}, "whole number of squares"),
])
def test_a_cube_that_is_not_a_ccube2_is_refused(values, refused) -> None:
    with pytest.raises(ValueError, match=refused):
        Ccube2(**values)


def test_without_aruco2_a_ccube2_is_still_named_and_says_how_to_get_it(monkeypatch) -> None:
    """pyCamSet.Ccube2 stays the class; building one names the fix, and in
    words that fit a target with no marker_backend of its own."""
    import sys

    import pyCamSet

    monkeypatch.setitem(sys.modules, "aruco2", None)
    assert pyCamSet.Ccube2 is Ccube2
    with pytest.raises(ImportError) as missing:
        Ccube2(draw_res=(100, 100))
    message = str(missing.value)
    assert "CITATION.md" in message
    assert "Ccube2" in message


def test_a_whole_number_of_squares_written_as_a_float_builds() -> None:
    """A spec read back from JSON may carry 5.0 for 5."""
    cube = Ccube2(n_points=5.0, draw_res=(100, 100))
    assert cube.n_points == 5 and isinstance(cube.n_points, int)


def test_the_border_only_has_to_hold_the_band() -> None:
    """A band a quarter of a square deep just fits at 1/(2n+1)."""
    Ccube2(n_points=4, border_fraction=1 / 9)
    Ccube2(n_points=2, border_fraction=1 / 5)


@pytest.mark.parametrize("n_points", range(2, 13))
def test_the_minimum_border_quoted_is_one_that_is_accepted(n_points) -> None:
    """Typing the number the refusal quotes must not be refused again."""
    with pytest.raises(ValueError) as refused:
        Ccube2(n_points=n_points, border_fraction=0.01, aruco_dict="DICT_4X4_1000")
    quoted = float(re.search(r"= (\d\.\d{4});", str(refused.value)).group(1))
    Ccube2(n_points=n_points, border_fraction=quoted, draw_res=(100, 100))


def test_the_suggested_face_sizes_build_with_the_default_border() -> None:
    import inspect

    doc = inspect.getdoc(Ccube2.__init__)
    n_points_doc = doc.split(":param n_points:")[1].split(":param")[0]
    low, high = map(int, re.search(r"Suggested: (\d+)-(\d+)\.", n_points_doc).groups())
    for n_points in (low, high):
        Ccube2(n_points=n_points, draw_res=(100, 100))


def test_the_only_ceiling_a_ccube2_has_is_its_alphabet() -> None:
    """Every square of six faces: 6 x 2 x 2 = 24 fits 50 markers, 6 x 3 x 3
    = 54 does not."""
    Ccube2(n_points=2, border_fraction=0.2, aruco_dict="DICT_4X4_50")
    with pytest.raises(ValueError, match="needs 54 markers, six faces of 9"):
        Ccube2(n_points=3, border_fraction=0.2, aruco_dict="DICT_4X4_50")


# -- a face against aruco2's own renderer --------------------------------------


@pytest.mark.parametrize(
    "dict_name, n_points, bit_size, draw_res, border_fraction",
    [
        # square px = marker_bits * bit_size, divisible by marker_bits + 2
        # and by 4; the margin is then a whole number of pixels too.
        ("DICT_4X4_1000", 4, 24, 480, 0.2),
        ("DICT_5X5_1000", 3, 28, 560, 0.25),
        ("DICT_6X6_1000", 5, 16, 600, 0.2),
    ],
)
def test_every_face_texture_is_aruco2s_board_image(
        dict_name, n_points, bit_size, draw_res, border_fraction) -> None:
    """Each face, without its number, is exactly aruco2's image of a board
    with that face's ids, placed in the face's margin."""
    cube = Ccube2(length=40.0, n_points=n_points, aruco_dict=dict_name,
                  draw_res=(draw_res, draw_res), border_fraction=border_fraction)
    dict_int = int(getattr(aruco2, dict_name))
    band_px = dictionary_marker_bits(dict_int) * bit_size // 4
    margin_px = int(round(draw_res * border_fraction / 2))

    for k in range(6):
        reference = render_grid_board_image(
            cube.grid_size, dict_int, bit_size, cube.face_ids[k])
        texture = cube.face_texture(k, draw_board_id=False)
        top = margin_px - band_px
        crop = texture[top:top + reference.shape[0], top:top + reference.shape[1]]
        assert np.array_equal(crop, reference), f"face {k}"


def test_a_face_number_touches_no_marker_tab_or_corner_square() -> None:
    """The number is drawn only on white, inside the band below the board
    and clear of both its edges."""
    cube = Ccube2(length=40.0, n_points=5)
    px_per_m = cube.draw_res[0] / cube.length
    band = layout.band_depth(cube.square_size)
    for k in range(6):
        plain = cube.face_texture(k, draw_board_id=False)
        labelled = cube.face_texture(k)
        rows, cols = np.nonzero(plain != labelled)
        assert rows.size, f"face {k} has no number"
        assert np.all(plain[rows, cols] == 255), "drawn over black"
        # A pixel's footprint, [r, r+1), is strictly inside the band.
        assert rows.min() > (cube.length - cube.margin) * px_per_m
        assert rows.max() + 1 < (cube.length - cube.margin + band) * px_per_m
        # And under one standard square, away from the tabs either side.
        x_centre, _, _ = cube.face_label_anchor()
        half_square = cube.square_size / 2 * px_per_m
        assert np.all(np.abs(cols + 0.5 - x_centre * px_per_m) < half_square)


@pytest.mark.parametrize("n_points, border_fraction", [(5, 0.1), (5, 1 / 11), (8, 0.12), (2, 0.2)])
def test_every_face_is_detected_as_itself(n_points, border_fraction) -> None:
    """A face is found only under its own ids, with every corner, where the
    lattice puts it -- including at the smallest border the band fits in,
    where a tab meets the face's edge line."""
    cube = Ccube2(length=40.0, n_points=n_points, border_fraction=border_fraction)
    # detect_grid_board_corners reports the pixel-*corner* convention (see
    # its docstring), 0.5 px from _texture_pixels' pixel-*centre* one.
    expected = _texture_pixels(cube, _lattice(cube)) + 0.5
    for k, texture in enumerate(cube.textures):
        detection = cube.find_in_image(texture)
        assert detection.data_len == (n_points + 1) ** 2, f"face {k}"
        assert set(detection.keys[:, 0].tolist()) == {k}
        assert sorted(detection.keys[:, 1].tolist()) == list(range((n_points + 1) ** 2))
        error = np.abs(detection.image_points - expected[detection.keys[:, 1]])
        assert error.max() < 1.0, f"face {k}"


def test_a_face_number_does_not_move_a_detection() -> None:
    cube = Ccube2(length=40.0, n_points=5)
    for k in range(6):
        with_number = cube.find_in_image(cube.face_texture(k))
        without = cube.find_in_image(cube.face_texture(k, draw_board_id=False))
        assert np.array_equal(with_number.keys, without.keys)
        assert np.abs(with_number.image_points - without.image_points).max() < 0.05


def test_ccube2_refuses_an_image_it_cannot_read_exactly() -> None:
    cube = Ccube2(n_points=3, length=20.0, border_fraction=0.2)
    with pytest.raises(ValueError, match="uint8"):
        cube.find_in_image(cube.textures[0] / 2.0 + 0.25)


def test_ccube2_delegates_its_dtype_check_to_the_shared_aruco2_helper(monkeypatch) -> None:
    """Ccube2.find_in_image used to reimplement aruco2._as_uint8_image's
    uint8/integral-float convertibility check inline instead of calling it
    (round-3 review, P2). That duplication was invisible to
    test_ccube2_refuses_an_image_it_cannot_read_exactly: with the local
    block deleted outright, that test still passes, because
    detect_grid_board_corners's own call to the shared helper -- further
    down the same call, once face detection starts -- raises an
    equally-matching "uint8" message regardless of what Ccube2's own code
    did. This spies on the shared helper directly instead of on a message
    two different code paths can both produce, so it actually pins Ccube2
    to delegating rather than reimplementing."""
    calls = []
    original = ccube2_target_module._as_uint8_image

    def spy(image):
        calls.append(image)
        return original(image)

    monkeypatch.setattr(ccube2_target_module, "_as_uint8_image", spy)
    cube = Ccube2(n_points=3, length=20.0, border_fraction=0.2)
    with pytest.raises(ValueError, match="uint8"):
        cube.find_in_image(cube.textures[0] / 2.0 + 0.25)

    assert len(calls) == 1, (
        "Ccube2.find_in_image must validate the image's dtype by calling "
        "the shared aruco2._as_uint8_image helper directly (once, before "
        "any face is looked for), not by reimplementing its check inline."
    )


# -- a synthetic photograph of the cube ------------------------------------------


def _look_at(camera_centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera rotation and translation for a camera at
    ``camera_centre`` looking at the origin."""
    z = -camera_centre / np.linalg.norm(camera_centre)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z])
    return rotation, -rotation @ camera_centre


def _project(points: np.ndarray, intrinsic: np.ndarray, rotation, translation) -> np.ndarray:
    in_camera = points @ rotation.T + translation
    uvw = in_camera @ intrinsic.T
    return uvw[:, :2] / uvw[:, 2:]


@pytest.mark.parametrize("direction, facing", [((1.0, 1.0, 1.0), [0, 2, 3]),
                                               ((-1.0, 1.0, 1.0), [0, 3, 5]),
                                               ((1.0, -1.0, -1.0), [1, 2, 4])],
                         ids=["faces-0-2-3", "faces-0-3-5", "faces-1-2-4"])
def test_a_photographed_cube_is_detected_where_its_point_data_projects(direction, facing) -> None:
    """End to end: each face texture is warped into a pinhole image by the
    homography its face-local corners take to the projection of its
    ``point_data`` corners, three faces at a time. Every corner detected
    must be a visible face's, and land within 1 px of where its
    ``point_data`` projects -- which ties the texture, the ids and the
    ``[face, gid]`` keys to ``point_data``. The image is warped from
    ``point_data`` itself, so it cannot tell whether ``point_data`` folds the
    cube the right way; :func:`test_ccube2_folds_its_faces_as_a_ccube_does`
    is what checks that."""
    cube = Ccube2(length=40.0, n_points=5, draw_res=(800, 800))
    n = cube.n_points
    width, height, focal = 1600, 1200, 1500.0
    intrinsic = np.array([[focal, 0, (width - 1) / 2],
                          [0, focal, (height - 1) / 2],
                          [0, 0, 1.0]])
    camera_centre = np.asarray(direction) / np.linalg.norm(direction) * 0.14
    rotation, translation = _look_at(camera_centre)

    # Drawn at 4x and averaged down, so a face is not aliased by the warp.
    # A pixel centre at u is at 4u + 1.5 in the supersampled image.
    ss = 4
    supersampled = np.array([[ss, 0, (ss - 1) / 2], [0, ss, (ss - 1) / 2], [0, 0, 1]]) @ intrinsic
    image = np.full((height * ss, width * ss), 255, dtype=np.uint8)

    texture_corners = _texture_pixels(cube, _lattice(cube))
    outer = [0, n, (n + 1) ** 2 - 1, n * (n + 1)]
    visible = []
    for k, face in enumerate(cube.point_data):
        outward = np.cross(face[1] - face[0], face[n + 1] - face[0])
        if outward @ face.mean(axis=0) < 0:
            outward = -outward
        if outward @ (camera_centre - face.mean(axis=0)) <= 0:
            continue  # facing away
        visible.append(k)
        homography = cv2.getPerspectiveTransform(
            texture_corners[outer].astype(np.float32),
            _project(face[outer], supersampled, rotation, translation).astype(np.float32))
        size = (width * ss, height * ss)
        warped = cv2.warpPerspective(
            cube.textures[k], homography, size, flags=cv2.INTER_LINEAR, borderValue=255)
        inside = cv2.warpPerspective(
            np.full_like(cube.textures[k], 255), homography, size,
            flags=cv2.INTER_NEAREST, borderValue=0) > 0
        image[inside] = warped[inside]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    assert visible == facing  # between them, the three views see every face

    detection = cube.find_in_image(image)
    keys = detection.keys
    assert set(keys[:, 0].tolist()) == set(visible)
    for k in visible:
        assert int(np.sum(keys[:, 0] == k)) == (n + 1) ** 2, f"face {k}"
    expected = _project(cube.point_data[keys[:, 0], keys[:, 1]], intrinsic, rotation, translation)
    error = np.linalg.norm(detection.image_points - expected, axis=1)
    assert error.max() < 1.0


# -- printing ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", EXPORT_KINDS)
def test_ccube2_save_printable_all_kinds(tmp_path: Path, kind: str) -> None:
    if kind == "pdf_vector":
        _skip_without_cairo()
    cube = Ccube2(n_points=4, length=20.0, border_fraction=0.15, draw_res=(400, 400))
    out = cube.save_printable(tmp_path / f"cube.{kind}", kind=kind)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 1024
    if kind.startswith("pdf_"):
        # The vector PDF is drawn, never a raster embedded in a page; the
        # raster one is exactly that.
        embeds_raster = re.search(rb"/Subtype\s*/Image", Path(out).read_bytes())
        assert bool(embeds_raster) == (kind == "pdf_raster")


def test_ccube2_prints_one_face_per_page_into_the_file_asked_for(tmp_path: Path) -> None:
    cube = Ccube2(n_points=4, length=20.0, border_fraction=0.15, draw_res=(200, 200))
    out = cube.save_printable(tmp_path / "faces.pdf_raster", kind="pdf_raster",
                              individual_faces=True)
    assert out == (tmp_path / "faces.pdf").resolve()
    pages = re.findall(rb"/Type\s*/Page(?!s)", out.read_bytes())
    assert len(pages) == 6
    assert list(tmp_path.iterdir()) == [out], "nothing written beside it"


def test_a_vector_pdf_of_single_faces_says_it_is_raster(tmp_path: Path, caplog) -> None:
    """Face-per-page pages come from the textures, so asking for a vector
    PDF of them must not silently give a raster one."""
    import logging

    cube = Ccube2(n_points=4, length=20.0, border_fraction=0.15, draw_res=(400, 400))
    with caplog.at_level(logging.WARNING, logger="pyCamSet.calibration_targets.ccube2.target"):
        cube.save_printable(tmp_path / "faces.pdf", kind="pdf_vector", individual_faces=True)
    assert any("raster" in record.getMessage() and record.levelno == logging.WARNING
               for record in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pyCamSet.calibration_targets.ccube2.target"):
        cube.save_printable(tmp_path / "faces2.pdf", kind="pdf_raster", individual_faces=True)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("draw_cut_outline, draw_board_ids",
                         [(False, False), (False, True), (True, False), (True, True)])
def test_a_raster_pdf_draws_the_outline_and_numbers_it_is_asked_to(
        tmp_path: Path, monkeypatch, draw_cut_outline, draw_board_ids) -> None:
    """Create Target offers both boxes for every format, so unticking one
    for a PDF must change the PDF, not only the SVG."""
    cube = Ccube2(n_points=4, length=20.0, border_fraction=0.15, draw_res=(200, 200))
    drawn = []
    draw_net = cube.faceData.draw_net

    def spy(textures, forms):
        drawn.extend(textures)
        return draw_net(textures, forms)

    monkeypatch.setattr(cube.faceData, "draw_net", spy)
    cube.save_printable(tmp_path / "cube.pdf", kind="pdf_raster",
                        draw_cut_outline=draw_cut_outline, draw_board_ids=draw_board_ids)
    assert len(drawn) == 6
    for k, texture in enumerate(drawn):
        expected = cube.face_texture(k, draw_board_id=draw_board_ids,
                                     draw_edge_line=draw_cut_outline)
        assert np.array_equal(texture, expected), f"face {k}"
    # And the flags are told apart: an outline and a number each show.
    assert (drawn[0][0, :] == 0).all() == draw_cut_outline
    unnumbered = cube.face_texture(1, draw_board_id=False, draw_edge_line=draw_cut_outline)
    numbered = not np.array_equal(drawn[1], unnumbered)
    assert numbered == draw_board_ids


def test_a_vector_pdf_draws_the_outline_and_numbers_it_is_asked_to(
        tmp_path: Path, monkeypatch) -> None:
    _skip_without_cairo()
    cube = Ccube2(n_points=4, length=20.0, border_fraction=0.15, draw_res=(200, 200))
    asked = {}
    save_to_svg = cube.save_to_svg

    def spy(*args, **kwargs):
        asked.update(kwargs)
        return save_to_svg(*args, **kwargs)

    monkeypatch.setattr(cube, "save_to_svg", spy)
    out = cube.save_printable(tmp_path / "cube.pdf", kind="pdf_vector",
                              draw_cut_outline=False, draw_board_ids=False)
    assert Path(out).exists()
    assert asked["draw_cut_outline"] is False
    assert asked["draw_board_ids"] is False


def test_a_face_per_page_pdf_draws_the_outline_and_numbers_it_is_asked_to(
        tmp_path: Path, monkeypatch) -> None:
    from pyCamSet.calibration_targets.ccube2 import target as ccube2_target

    cube = Ccube2(n_points=4, length=20.0, border_fraction=0.15, draw_res=(200, 200))
    pages = []
    fromarray = ccube2_target.Image.fromarray

    def spy(array, *args, **kwargs):
        pages.append(np.array(array))
        return fromarray(array, *args, **kwargs)

    monkeypatch.setattr(ccube2_target.Image, "fromarray", spy)
    cube.save_printable(tmp_path / "faces.pdf", kind="pdf_raster", individual_faces=True,
                        border_width=0.0, draw_cut_outline=False, draw_board_ids=False)
    assert len(pages) == 6
    for k, page in enumerate(pages):
        expected = cube.face_texture(k, draw_board_id=False, draw_edge_line=False)
        assert np.array_equal(page, expected), f"face {k}"


@pytest.mark.parametrize("draw_res", [100, 200, 333])
def test_a_small_face_texture_is_still_a_face(draw_res) -> None:
    """A draw_res below 1 / line_fraction rounds the edge line to no pixels,
    which Ccube's helper turns into a solid black face."""
    cube = Ccube2(n_points=4, length=20.0, border_fraction=0.15,
                  draw_res=(draw_res, draw_res))
    for k, texture in enumerate(cube.textures):
        black = float(np.mean(texture == 0))
        assert 0.3 < black < 0.7, f"face {k}: {black:.2f} black"
        # The edge line is still drawn, one pixel wide.
        assert np.all(texture[:, 0] == 0) and np.all(texture[-1, :] == 0)


def test_ccube2_svg_is_true_vector_at_millimetre_scale(tmp_path: Path) -> None:
    """One path per face and no embedded raster; the page is the 3 x 4 face
    net plus the border, in mm."""
    cube = Ccube2(n_points=4, length=30.0, border_fraction=0.15)
    text = cube.save_to_svg(tmp_path / "cube.svg", border_width=10.0).read_text(encoding="utf-8")
    assert "<image" not in text
    assert "base64" not in text
    assert text.count("<path") == 6
    assert text.count('fill-rule="nonzero"') == 6
    w_mm, h_mm = _svg_size_mm(text)
    assert w_mm == pytest.approx(3 * 30.0 + 2 * 10.0, abs=1e-6)
    assert h_mm == pytest.approx(4 * 30.0 + 2 * 10.0, abs=1e-6)


@pytest.mark.parametrize("px_per_mm", [12.0, 300 / 25.4])
def test_a_printed_net_is_detected_face_by_face_with_its_numbers_on(tmp_path: Path, px_per_mm) -> None:
    """The SVG, rasterised by cairo with the cut outline and face numbers
    drawn, gives every corner of all six faces, each where the net puts it."""
    _skip_without_cairo()
    cube = Ccube2(n_points=5, length=60.0)
    border_mm = 5.0
    svg = cube.save_to_svg(tmp_path / "cube.svg", border_width=border_mm)
    assert "<text" in svg.read_text(encoding="utf-8")
    image = _rasterise_svg(svg, px_per_mm)

    detection = cube.find_in_image(image)
    keys = detection.keys
    for k in range(6):
        assert int(np.sum(keys[:, 0] == k)) == 36, f"face {k}"

    outline = np.array([[0, 0], [cube.length, 0], [cube.length, cube.length], [0, cube.length]])
    net_min = np.vstack([cube.apply_affine_xy(outline, cube.net_affine_for_face(k))
                         for k in range(6)]).min(axis=0)
    lattice = _lattice(cube)
    expected = np.array([
        cube.apply_affine_xy(lattice[[gid]], cube.net_affine_for_face(face))[0]
        for face, gid in keys])
    # detect_grid_board_corners' pixel-*corner* convention, not the raster's
    # own pixel-*centre* one (see _texture_pixels) -- so +0.5, not -0.5.
    expected = (expected - net_min + border_mm / 1000) * 1000 * px_per_mm + 0.5
    assert np.abs(detection.image_points - expected).max() < 1.0


def test_a_printed_face_number_touches_no_marker_tab_or_corner_square(tmp_path: Path) -> None:
    """The SVG's numbers, whatever font cairo substitutes for Arial, land only
    on white, each in the band below its own face's board, under the square
    :meth:`Ccube2.face_label_anchor` picks."""
    _skip_without_cairo()
    cube = Ccube2(n_points=5, length=60.0)
    px_per_mm = 20.0
    plain = _rasterise_svg(cube.save_to_svg(
        tmp_path / "plain.svg", border_width=0.0, draw_cut_outline=False,
        draw_board_ids=False), px_per_mm)
    numbered = _rasterise_svg(cube.save_to_svg(
        tmp_path / "numbered.svg", border_width=0.0, draw_cut_outline=False,
        draw_board_ids=True), px_per_mm)
    assert plain.shape == numbered.shape

    rows, cols = np.nonzero(np.abs(plain.astype(int) - numbered.astype(int)) > 0)
    assert rows.size
    assert np.all(plain[rows, cols] == 255), "a number was drawn over black"

    outline = np.array([[0, 0], [cube.length, 0], [cube.length, cube.length], [0, cube.length]])
    affines = [cube.net_affine_for_face(k) for k in range(6)]
    net_min = np.vstack([cube.apply_affine_xy(outline, a) for a in affines]).min(axis=0)
    # Pixel centres, in net metres, taken back into each face's own frame.
    centres = np.stack([cols + 0.5, rows + 0.5], axis=-1) / (px_per_mm * 1000) + net_min
    x_centre, _, _ = cube.face_label_anchor()
    band = layout.band_depth(cube.square_size)
    pixel = 1 / (px_per_mm * 1000)  # anti-aliased edges reach half a pixel out
    on_a_face = np.zeros(rows.size, dtype=bool)
    for k, a in enumerate(affines):
        local = cube.apply_affine_xy(centres, np.linalg.inv(a))
        inside = np.all((local >= 0) & (local <= cube.length), axis=1)
        on_a_face |= inside
        local = local[inside]
        assert local.size, f"face {k} has no number"
        assert np.all(local[:, 1] > cube.length - cube.margin + pixel), f"face {k}: on the board"
        assert np.all(local[:, 1] < cube.length - cube.margin + band - pixel), f"face {k}: off the band"
        assert np.all(np.abs(local[:, 0] - x_centre) < cube.square_size / 2 - pixel), \
            f"face {k}: beside its square, where a tab or corner square is"
    assert on_a_face.all()


def test_the_vector_net_is_the_raster_net(tmp_path: Path) -> None:
    """The SVG places every face as the raster PDF's ``draw_net`` does, so
    folding either gives the same cube.

    Compared face by face, inside each face's edge line (which the two draw
    differently: a stroke centred on the cut against a line inside the
    texture). ``draw_net`` rotates a face about pixel indices rather than
    pixel edges, so a quarter-turned face lands one pixel over from where
    the vector puts it; each face is therefore allowed a one-pixel shift,
    and must then match exactly -- while a face in the wrong place or turned
    the wrong way differs in a sixth or more of its pixels."""
    _skip_without_cairo()
    face_px = 600
    cube = Ccube2(n_points=5, length=60.0, draw_res=(face_px, face_px))
    svg = cube.save_to_svg(tmp_path / "cube.svg", border_width=0.0,
                           draw_cut_outline=False, draw_board_ids=False)
    px_per_m = face_px / cube.length
    vector = _rasterise_svg(svg, px_per_m / 1000)  # face_px a face, as the textures are
    textures = [cube.face_texture(k, draw_board_id=False) for k in range(6)]
    raster = cube.faceData.draw_net(textures, NET_FORMS)
    assert vector.shape == raster.shape

    outline = np.array([[0, 0], [cube.length, 0], [cube.length, cube.length], [0, cube.length]])
    placed = [cube.apply_affine_xy(outline, cube.net_affine_for_face(k)) for k in range(6)]
    net_min = np.vstack(placed).min(axis=0)
    inset = 5  # clear of the edge line and of the one-pixel shift
    for k, corners in enumerate(placed):
        x0, y0 = np.round((corners.min(axis=0) - net_min) * px_per_m).astype(int)
        window = np.s_[y0 + inset:y0 + face_px - inset, x0 + inset:x0 + face_px - inset]
        drawn = vector[window].astype(int)
        mismatch = {}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted = raster[y0 + inset + dy:y0 + face_px - inset + dy,
                                 x0 + inset + dx:x0 + face_px - inset + dx].astype(int)
                mismatch[dy, dx] = float(np.mean(np.abs(drawn - shifted) > 127))
        assert min(mismatch.values()) == 0.0, f"face {k}: {mismatch}"
        if np.allclose(cube.net_affine_for_face(k)[:2, :2], np.eye(2)):
            assert mismatch[0, 0] == 0.0, f"unrotated face {k} moved"
        # And the comparison can tell: the same face half-turned does not match.
        half_turned = np.rot90(raster[window].astype(int), 2)
        assert np.mean(np.abs(drawn - half_turned) > 127) > 0.1, f"face {k}"


def test_generate_ccube2_target_builds_and_returns_saved_path(tmp_path: Path) -> None:
    from pyCamSet.calibration_targets.ccube2.generate import (
        build_ccube2, default_output_name, generate_ccube2_target,
    )

    cube = build_ccube2(n_points=4, length=20, border_fraction=0.15)
    assert cube.point_data.shape == (6, 25, 3)
    assert default_output_name(4, 20, "pdf_vector") == "ccube2_4points_20mm.pdf"

    built, saved = generate_ccube2_target(
        n_points=5,
        length=20,
        output_dir=tmp_path,
        file_name="nested/ccube2.txt",
        export_kind="svg",
    )
    assert isinstance(built, Ccube2)
    assert saved == (tmp_path / "nested/ccube2.svg").resolve()
    assert saved.exists()
    assert saved.stat().st_size > 0
