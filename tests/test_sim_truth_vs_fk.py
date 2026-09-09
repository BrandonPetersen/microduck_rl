"""Quantify the gap between MuJoCo's camera and the camera the robot's FK believes in.

The two MJCF assets disagree: the sim renders from pos="0.0155 -9.13778e-05 -0.0733"
(robot_allcollisions.xml) and microduck/kinematics/assets/alpha/robot_walk.xml:66 says
pos="0.01175 0 -0.0735". This test prints the resulting world-frame error so rung 4 has a
number to expect instead of a surprise.
"""

from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.sim.body_server import Body, World

SIM_CAM_IN_HEAD = np.array([0.0155, -9.13778e-05, -0.0733])
KINEMATICS_CAM_IN_HEAD = np.array([0.01175, 0.0, -0.0735])


def test_report_the_camera_asset_offset(capsys):
    root = Path(__file__).parent.parent
    # `count`, not `ducks`; and pose + forward before reading, since World.__init__ does neither.
    world = World(root / "src/mjlab_microduck/robot/microduck/scene_vslam.xml", count=1)
    body = Body(world, index=0)
    key = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(world.model, world.data, key)
    mujoco.mj_forward(world.model, world.data)
    truth = body.truth()
    offset_m = float(np.linalg.norm(SIM_CAM_IN_HEAD - KINEMATICS_CAM_IN_HEAD))
    with capsys.disabled():
        print(f"\ncamera asset offset (head frame): {offset_m * 1000:.2f} mm")
        print(f"true camera world pos: {np.round(truth['cam_pos'], 4)}")
        print(f"true trunk world pos:  {np.round(truth['trunk'], 4)}")
    assert offset_m < 0.005, "the two assets have diverged further than the known 3.75 mm"
