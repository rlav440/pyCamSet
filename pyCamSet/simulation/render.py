"""
A cube target's faces drawn into a pinhole camera's image.

Each visible face is two layers: the solid's own face, in the plastic's
colour, and the paper pasted onto it -- the target's face texture, placed by
the face's :class:`~.imperfection.PastingErrors` and trimmed to the face, as
a person trims paper that overhangs an edge.  Both are lit by one distant
light, so faces turned away from it are darker, as they are in a real shot.

The drawing is supersampled and averaged down, so edges are anti-aliased the
way a sensor's pixels average the light that falls on them.  The geometry is
the one ``tests/test_simulation.py`` checks detection against: a texture's pixel
centres are at ``(j + 0.5) / px_per_m`` in its face's frame, and a camera's
pixel centres are at integer coordinates of its matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .imperfection import PastingErrors, face_to_world
from .rig import SimCamera

#: The steepest view of a face still counted as readable: past about 75
#: degrees a face is too foreshortened to read.
READABLE_ANGLE_DEG = 75.0


@dataclass
class Scene:
    """
    What surrounds the target.

    :param light: the direction light comes *from*, in world coordinates.
    :param ambient: the share of light that reaches every face regardless.
    :param plastic: the core's albedo, where paper does not cover it.
    :param paper: white paper's albedo; the texture's black is ``ink``.
    :param ink: printed black's albedo.
    :param background: the mean albedo behind the target.
    :param background_texture: how much the background varies, so it is not
        a flat field a detector could never be confused by.
    :param supersample: how many samples per pixel each way.
    """

    light: tuple[float, float, float] = (0.4, -0.3, 1.0)
    ambient: float = 0.35
    plastic: float = 0.55
    paper: float = 0.92
    ink: float = 0.06
    background: float = 0.3
    background_texture: float = 0.12
    supersample: int = 3


def render_view(target, camera: SimCamera, pose: np.ndarray, errors: PastingErrors,
                scene: Scene | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Draw the target, held at ``pose``, as ``camera``'s ideal pinhole image.

    :param target: a cube target with ``textures``, ``faceData`` and
        ``length`` (in metres).
    :param camera: the camera.
    :param pose: the target-to-world 4x4 transform.
    :param errors: how the target's faces were pasted.
    :param scene: the lighting and the background; the defaults when not given.
    :param rng: for the background's texture.
    :return: a float32 radiance image, ``(height, width)``, 0 to 1.
    """
    scene = Scene() if scene is None else scene
    rng = np.random.default_rng(0) if rng is None else rng
    width, height = camera.res
    ss = int(scene.supersample)
    # A pixel centre at u is at ss * u + (ss - 1) / 2 in the supersampled image.
    upscale = np.array([[ss, 0, (ss - 1) / 2.0], [0, ss, (ss - 1) / 2.0], [0, 0, 1.0]])
    matrix = upscale @ camera.intrinsic
    size = (width * ss, height * ss)

    canvas = _background((height * ss, width * ss), scene, rng)
    world_to_camera = camera.extrinsic @ pose
    light = np.asarray(scene.light, dtype=float)
    light /= np.linalg.norm(light)
    length = float(target.length)
    square = np.array([[0.0, 0.0], [length, 0.0], [length, length], [0.0, length]])

    faces = []
    for face in range(len(target.faceData.ur_ftform)):
        corners = _to(world_to_camera, face_to_world(target, face, square))
        centre = corners.mean(axis=0)
        normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
        cube_centre = world_to_camera[:3, 3]           # the target's origin is its centre
        if normal @ (centre - cube_centre) < 0:
            normal = -normal
        if normal @ (-centre) <= 0:                    # turned away from the camera
            continue
        world_normal = camera.extrinsic[:3, :3].T @ (normal / np.linalg.norm(normal))
        shade = scene.ambient + (1.0 - scene.ambient) * max(0.0, float(world_normal @ light))
        faces.append((centre[2], face, corners, shade))

    for _, face, corners, shade in sorted(faces, key=lambda item: -item[0]):
        face_px = _project(corners, matrix)
        # The solid's face, in plastic.
        face_mask = np.zeros((size[1], size[0]), dtype=np.float32)
        # Hard-edged here; supersampling averages the edge down.
        cv2.fillConvexPoly(face_mask, np.round(face_px * 16).astype(np.int32), 1.0,
                           lineType=cv2.LINE_8, shift=4)
        canvas = canvas * (1.0 - face_mask) + scene.plastic * shade * face_mask

        # The paper on it: texture pixels to paper, paper to face, face to image.
        texture = np.asarray(target.textures[face], dtype=np.float32) / 255.0
        albedo = scene.ink + (scene.paper - scene.ink) * texture
        res_y, res_x = texture.shape[:2]
        texture_corners = square * (np.array([res_x, res_y]) / length) - 0.5
        on_face = np.c_[square, np.ones(4)] @ errors.paper_to_face(face, length).T
        paper_px = _project(_to(world_to_camera, face_to_world(target, face, on_face[:, :2])),
                            matrix)
        homography = cv2.getPerspectiveTransform(texture_corners.astype(np.float32),
                                                 paper_px.astype(np.float32))
        warped = cv2.warpPerspective(albedo, homography, size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        cover = cv2.warpPerspective(np.ones_like(albedo), homography, size,
                                    flags=cv2.INTER_LINEAR, borderValue=0.0)
        alpha = cover * face_mask                      # trimmed at the face's edge
        canvas = canvas * (1.0 - alpha) + warped * shade * alpha

    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)


def project_points(points_world: np.ndarray, camera: SimCamera) -> np.ndarray:
    """Where world points land in ``camera``'s distorted image."""
    rvec = cv2.Rodrigues(camera.extrinsic[:3, :3])[0]
    image, _ = cv2.projectPoints(np.asarray(points_world, dtype=np.float64).reshape(-1, 1, 3),
                                 rvec, camera.extrinsic[:3, 3], camera.intrinsic,
                                 camera.distortion)
    return image.reshape(-1, 2)


def readable_pixels(true_points: np.ndarray, n_faces: int, camera: SimCamera,
                    pose: np.ndarray) -> np.ndarray:
    """
    Where a target's points land in a camera's frame, on the faces it can read.

    A face counts when it is turned to within :data:`READABLE_ANGLE_DEG` of
    the camera (its outward normal, from the target's origin through the
    face's middle, against the direction to the camera) and a point counts
    when it lands inside the image.

    :param true_points: the target's points in its own frame, shaped like
        ``point_data``.
    :param n_faces: how many faces the points are split over.
    :param camera: the camera.
    :param pose: the target-to-world 4x4 transform.
    :return: an ``(N, 2)`` array of pixel positions.
    """
    flat = np.asarray(true_points, float).reshape(n_faces, -1, 3)
    width, height = camera.res
    found = []
    for face_points in flat:
        world = face_points @ pose[:3, :3].T + pose[:3, 3]
        middle = world.mean(axis=0)
        normal = middle - pose[:3, 3]                  # a cube face's normal points out
        view = camera.centre - middle
        cosine = normal @ view / (np.linalg.norm(normal) * np.linalg.norm(view))
        if cosine <= np.cos(np.radians(READABLE_ANGLE_DEG)):
            continue
        uv = project_points(world, camera)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        found.append(uv[inside])
    return np.concatenate(found) if found else np.zeros((0, 2))


def _to(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _project(points_camera: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    uvw = points_camera @ matrix.T
    return uvw[:, :2] / uvw[:, 2:]


def _background(shape, scene: Scene, rng: np.random.Generator) -> np.ndarray:
    """A softly lit, mottled backdrop: a gradient plus low-frequency texture."""
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    gradient = 0.15 * (xs / width - 0.5) + 0.1 * (ys / height - 0.5)
    coarse = rng.normal(0.0, 1.0, (max(2, height // 64), max(2, width // 64))).astype(np.float32)
    mottle = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(scene.background * (1.0 + gradient)
                   + scene.background_texture * scene.background * mottle, 0.0, 1.0)
