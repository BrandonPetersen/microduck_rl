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
