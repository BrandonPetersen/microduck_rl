"""How much the depth pass costs, so a regression in it is visible.

Rendered against `scene_vslam.xml` (151 geoms) rather than a synthetic one-box fixture: the render
cost is dominated by copying the scene in `update_scene`, not by shading it, so an empty world
measures a floor that is roughly half the real cost and would understate the number a later task's
real-time verdict gets quoted against. The docstring in camera.py claims 12.2 ms for one 640x360
RGB frame in a real scene; a depth pass adds only ~2 ms on top of that, because the scene copy -- not
the render -- is what's expensive.
"""

import threading
import time
from pathlib import Path

import mujoco

from mjlab_microduck.sim.camera import Camera

SCENE_VSLAM = (
    Path(__file__).parent.parent / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene_vslam.xml"
)


class _World:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.lock = threading.Lock()


def test_rgb_plus_depth_render_cost_is_reported(capsys):
    model = mujoco.MjModel.from_xml_path(str(SCENE_VSLAM))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    world = _World(model, data)
    cam = Camera(model, "head_camera", width=640, height=360)
    cam.render(world)  # warm the GL context; the first render pays for setup
    n = 10
    start = time.perf_counter()
    for _ in range(n):
        cam.render(world)
    ms = (time.perf_counter() - start) / n * 1000.0
    with capsys.disabled():
        print(
            f"\nRGB+depth at 640x360 ({SCENE_VSLAM.name}): {ms:.1f} ms/frame "
            f"({ms * 15 / 1000:.2f} of a core at 15 fps)"
        )
    assert ms < 40.0, f"{ms:.1f} ms/frame is too slow to keep the sim in real time"
