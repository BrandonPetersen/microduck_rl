"""Quantify the gap between MuJoCo's camera and the camera the robot's FK believes in.

The two MJCF assets disagree: the sim renders from pos="0.0155 -9.13778e-05 -0.0733"
(robot_allcollisions.xml) and microduck/kinematics/assets/alpha/robot_walk.xml:66 says
pos="0.01175 0 -0.0735". This test prints the resulting world-frame error so rung 4 has a
number to expect instead of a surprise.

The known offset is a **3.76 mm** norm; its x-component alone is 3.75 mm, which is where the
similar-looking "3.75 mm" figure quoted elsewhere comes from -- the norm is what this test and
the rest of the branch assert against.
"""

from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.sim.body_server import Body, World

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
    # Read the sim's camera-in-head offset off the model itself, not a copied-in literal, so an
    # edit to the rendered MJCF asset's camera pos trips this test instead of comparing two
    # constants that can never disagree.
    cam_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    sim_cam_in_head = np.array(world.model.cam_pos[cam_id])
    offset_m = float(np.linalg.norm(sim_cam_in_head - KINEMATICS_CAM_IN_HEAD))
    with capsys.disabled():
        print(f"\ncamera asset offset (head frame): {offset_m * 1000:.2f} mm")
        print(f"true camera world pos: {np.round(truth['cam_pos'], 4)}")
        print(f"true trunk world pos:  {np.round(truth['trunk'], 4)}")
    assert offset_m < 0.005, "the two assets have diverged further than the known 3.76 mm"
