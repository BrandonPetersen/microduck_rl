"""How much the depth pass costs, so a regression in it is visible.

The docstring in camera.py claims 12.2 ms for one 640x360 RGB frame. A depth pass is comparable,
so a camera should cost ~25 ms a frame. At 15 fps that is 0.4 of a core -- which matters, because
the daemon's health gate fails below 45 of 50 Hz.
"""

import threading
import time

import mujoco
import numpy as np

from mjlab_microduck.sim.camera import Camera


class _World:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.lock = threading.Lock()
        mujoco.mj_forward(model, self.data)


def test_rgb_plus_depth_render_cost_is_reported(capsys):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><statistic extent="1.0"/><visual><map znear="0.02" zfar="20"/></visual>'
        '<worldbody><camera name="head_camera" pos="0 0 0" quat="1 0 0 0"/>'
        '<geom type="box" size="5 5 0.01" pos="0 0 -2"/></worldbody></mujoco>'
    )
    world = _World(model)
    cam = Camera(model, "head_camera", width=640, height=360)
    cam.render(world)  # warm the GL context; the first render pays for setup
    n = 10
    start = time.perf_counter()
    for _ in range(n):
        cam.render(world)
    ms = (time.perf_counter() - start) / n * 1000.0
    with capsys.disabled():
        print(f"\nRGB+depth at 640x360: {ms:.1f} ms/frame ({ms * 15 / 1000:.2f} of a core at 15 fps)")
    assert ms < 100.0, f"{ms:.1f} ms/frame is too slow to keep the sim in real time"
