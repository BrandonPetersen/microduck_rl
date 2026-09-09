"""Ground-truth depth from MuJoCo, checked against geometry we wrote ourselves.

MuJoCo 3.10's Renderer wrapper linearises the depth buffer and returns float32 METRES, so there is
nothing to convert -- but it returns depth along the OPTICAL AXIS, not ray range, and it returns
exactly zfar where no geometry was hit. Both are pinned here.
"""

import threading

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.camera import ZFAR_M, Camera

# A camera at the origin looking down -z at a wall filling the view. The box is CENTRED at 2.0 m
# with a half-thickness of 0.01, so the surface the camera actually sees -- and therefore the depth
# MuJoCo reports -- is at 1.99 m, not 2.00 m. Measured on mujoco 3.10.0: 1.9900000095367432.
# Getting this wrong is how a correct depth renderer gets mistaken for a 10 mm bias.
WALL_CENTRE_M = 2.0
WALL_HALF_THICKNESS_M = 0.01
WALL_FACE_M = WALL_CENTRE_M - WALL_HALF_THICKNESS_M  # 1.99
SCENE = f"""
<mujoco>
  <statistic extent="1.0"/>
  <visual><map znear="0.02" zfar="{ZFAR_M}"/></visual>
  <worldbody>
    <camera name="head_camera" pos="0 0 0" quat="1 0 0 0"/>
    <geom name="wall" type="box" size="5 5 0.01" pos="0 0 -{WALL_CENTRE_M}"/>
  </worldbody>
</mujoco>
"""


# The same wall, seen OBLIQUELY. A wall square to the camera reads one depth everywhere, so it
# cannot distinguish one ray-construction convention from another; tilting the camera makes depth a
# function of the ray direction, which is exactly what the convention decides.
TILT_DEG = 35.0
TILTED_SCENE = f"""
<mujoco>
  <statistic extent="1.0"/>
  <visual><map znear="0.02" zfar="{ZFAR_M}"/></visual>
  <worldbody>
    <camera name="head_camera" pos="0 0 0" euler="{TILT_DEG} 0 0"/>
    <geom name="wall" type="box" size="5 5 0.01" pos="0 0 -{WALL_CENTRE_M}"/>
  </worldbody>
</mujoco>
"""


class _World:
    """The two attributes Camera.render touches."""

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.lock = threading.Lock()
        mujoco.mj_forward(model, self.data)


def _rendered():
    model = mujoco.MjModel.from_xml_string(SCENE)
    world = _World(model)
    cam = Camera(model, "head_camera", width=64, height=64)
    cam.render(world)
    return cam, world


def test_depth_is_metres_and_matches_the_wall_we_placed():
    cam, _ = _rendered()
    got = cam.wait_frame(after_seq=-1, timeout=1.0)
    assert got is not None
    _seq, _sim_time, _mono, _uyvy, depth = got
    assert depth.shape == (64, 64)
    assert depth.dtype == np.float32
    centre = float(depth[32, 32])
    # 5 mm is the spec's rung-0 acceptance tolerance; the real residual is ~1e-8 m.
    assert abs(centre - WALL_FACE_M) < 0.005, (
        f"centre depth {centre} m, wall FACE at {WALL_FACE_M} m "
        f"(box centred at {WALL_CENTRE_M} m, half-thickness {WALL_HALF_THICKNESS_M} m)"
    )


def test_depth_is_along_the_optical_axis_not_ray_range():
    """A flat wall square to the camera reads the SAME depth at the centre and at the corner.
    Ray range would grow toward the corner by 1/cos(theta)."""
    cam, _ = _rendered()
    _seq, _st, _mn, _uyvy, depth = cam.wait_frame(after_seq=-1, timeout=1.0)
    assert abs(float(depth[2, 2]) - float(depth[32, 32])) < 0.005


def test_background_reads_exactly_zfar():
    model = mujoco.MjModel.from_xml_string(
        SCENE.replace(f'<geom name="wall" type="box" size="5 5 0.01" pos="0 0 -{WALL_CENTRE_M}"/>', "")
    )
    world = _World(model)
    cam = Camera(model, "head_camera", width=32, height=32)
    cam.render(world)
    _seq, _st, _mn, _uyvy, depth = cam.wait_frame(after_seq=-1, timeout=1.0)
    assert float(depth[16, 16]) == pytest.approx(ZFAR_M, rel=1e-3)


def test_render_advances_seq_and_carries_sim_time():
    cam, world = _rendered()
    first = cam.wait_frame(after_seq=-1, timeout=1.0)
    world.data.time = 1.25
    cam.render(world)
    second = cam.wait_frame(after_seq=first[0], timeout=1.0)
    assert second[0] == first[0] + 1
    assert second[1] == pytest.approx(1.25)
    assert second[2] > 0


def test_rgb_and_depth_come_from_the_same_render():
    """Same seq, and the UYVY payload is the documented size."""
    cam, _ = _rendered()
    seq, _st, _mn, uyvy, depth = cam.wait_frame(after_seq=-1, timeout=1.0)
    assert len(uyvy) == 64 * 64 * 2
    assert depth.size == 64 * 64
    assert seq == 0


def test_frames_queue_up_so_a_slower_reader_loses_none():
    """Render several frames before reading any: every one comes back, oldest first.

    A single "latest" slot would hand back only the newest and silently drop the rest -- which is
    what port 7901 does, and the whole reason this bench stream exists."""
    cam, world = _rendered()  # already rendered seq 0
    for i in range(1, 4):
        world.data.time = 0.5 * i
        cam.render(world)
    seen, last = [], -1
    while True:
        got = cam.wait_frame(after_seq=last, timeout=0.2)
        if got is None:
            break
        seen.append((got[0], got[1]))
        last = got[0]
    assert [s for s, _ in seen] == [0, 1, 2, 3]
    assert [t for _, t in seen] == pytest.approx([0.0, 0.5, 1.0, 1.5])


def _predict_wall_depth(cam_pos, cam_mat, w, h, fovy_deg, offset):
    """Optical-axis depth of the plane z = -WALL_FACE_M, per pixel, for a given ray convention.

    `offset` is where inside pixel (j, i) the ray is taken to pass: 0.5 is the pixel CENTRE, 0.0 is
    its corner. `cam_mat` is MuJoCo's column convention (right, up, backward) in world, so optical
    (right, down, forward) is `M @ diag(1, -1, -1)`. Scaling the ray so its component along the
    optical axis is 1 makes the line parameter exactly the depth MuJoCo reports.
    """
    f = (h / 2) / np.tan(np.radians(fovy_deg) / 2)
    cx, cy = w / 2, h / 2
    jj, ii = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    M = np.asarray(cam_mat, dtype=np.float64).reshape(3, 3)
    right, down, fwd = M[:, 0], -M[:, 1], -M[:, 2]
    d = (((jj + offset - cx) / f)[..., None] * right
         + ((ii + offset - cy) / f)[..., None] * down + fwd)
    return (-WALL_FACE_M - cam_pos[2]) / d[..., 2]


@pytest.mark.parametrize("offset,wanted", [(0.5, True), (0.0, False)])
def test_rays_pass_through_pixel_centres_not_pixel_corners(offset, wanted):
    """MuJoCo samples at pixel CENTRES: column j's ray goes through j + 0.5.

    Nothing stated this before, and half a pixel is not a rounding detail. At the bench camera's
    fx = 434.5584 it is 0.5/434.5584 = 1.15e-3 rad = 0.066 deg, which where a ray grazes the floor
    near the horizon is up to 88 mm of depth -- seventeen times the spec's 5 mm rung-0 tolerance,
    on intrinsics that are exactly right. Measured on a real 667-frame capture of scene_vslam
    against analytic room geometry: 0.195 mm mean residual at j + 0.5 against 6.047 mm at j + 0.0,
    with a clean minimum at 0.5 across a 0.0/0.25/0.5/0.75/1.0 sweep.

    Synthetic and self-contained on purpose -- that capture is 881 MB and gitignored, so no test
    may depend on it.
    """
    w = h = 64
    model = mujoco.MjModel.from_xml_string(TILTED_SCENE)
    world = _World(model)
    cam = Camera(model, "head_camera", width=w, height=h)
    cam.render(world)
    _seq, _st, _mn, _uyvy, depth = cam.wait_frame(after_seq=-1, timeout=1.0)

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    pred = _predict_wall_depth(
        world.data.cam_xpos[cid], world.data.cam_xmat[cid], w, h,
        float(model.cam_fovy[cid]), offset,
    )
    # Every pixel hits the 10x10 m wall from ~2-4 m out, so nothing needs masking.
    assert np.all(depth < ZFAR_M), "some ray missed the wall; the fixture is wrong, not the code"
    worst = float(np.abs(depth - pred).max())

    if wanted:
        assert worst < 0.001, (
            f"rays through pixel centres (j + {offset}) should reproduce the plane, but the worst "
            f"residual is {worst * 1e3:.4f} mm"
        )
    else:
        # The wrong convention must be CAUGHT, not merely be worse. Measured on this fixture:
        # 0.0023 mm max at j + 0.5 against 21.7043 mm max (12.0805 mm mean) at j + 0.0, with the
        # 0.25 and 0.75 offsets at ~10.8 mm -- so both thresholds have an order of magnitude of
        # room and neither is tuned to the answer.
        assert worst > 0.002, (
            f"rays through pixel corners (j + {offset}) are only {worst * 1e3:.4f} mm off, so this "
            "fixture is too insensitive to guard the convention -- increase TILT_DEG or the FOV"
        )


def test_queue_is_bounded_and_a_drop_shows_as_a_seq_gap():
    from mjlab_microduck.sim.camera import BENCH_QUEUE

    cam, world = _rendered()
    for i in range(BENCH_QUEUE + 3):
        world.data.time = 0.1 * i
        cam.render(world)
    first = cam.wait_frame(after_seq=-1, timeout=0.5)
    # The oldest surviving frame is NOT seq 0: the overflow is visible, not silent.
    assert first[0] > 0
    assert cam.seq == BENCH_QUEUE + 3
