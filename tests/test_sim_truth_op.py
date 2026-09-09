"""The truth op: MuJoCo's own camera pose, not the robot's FK guess.

`{"op":"read"}` gives the TRUNK pose, so a consumer wanting the camera has to go through the
robot's forward kinematics -- which uses a different MJCF asset whose camera sits 3.76 mm away
from the one the sim renders from, and which publishes a frame rolled 90 deg from the rendered one.
Serving cam_xpos/cam_xmat sidesteps all of it.
"""

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.body_server import Body, Handler, World

SCENE = "src/mjlab_microduck/robot/microduck/scene_vslam.xml"


@pytest.fixture
def body():
    from pathlib import Path

    root = Path(__file__).parent.parent
    # `count`, not `ducks` (body_server.py:178). And World.__init__ does NOT call mj_forward, so
    # data.cam_xpos/cam_xmat are all zeros until we do -- posing the duck first, or the "camera is
    # above the trunk" check reads a collapsed model.
    world = World(root / SCENE, count=1)
    b = Body(world, index=0)
    key = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(world.model, world.data, key)
    mujoco.mj_forward(world.model, world.data)
    return b


def test_truth_reports_mujocos_own_camera_pose(body):
    truth = body.truth()
    cam = mujoco.mj_name2id(body.world.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    np.testing.assert_allclose(truth["cam_pos"], body.world.data.cam_xpos[cam], atol=1e-12)
    np.testing.assert_allclose(
        np.array(truth["cam_mat"]).reshape(3, 3), body.world.data.cam_xmat[cam].reshape(3, 3),
        atol=1e-12,
    )


def test_truth_carries_sim_time_and_the_trunk_pose(body):
    truth = body.truth()
    assert truth["sim_time"] == pytest.approx(float(body.world.data.time))
    assert len(truth["trunk"]) == 3
    assert len(truth["trunk_quat"]) == 4
    assert np.linalg.norm(truth["trunk_quat"]) == pytest.approx(1.0, abs=1e-6)


def test_truth_camera_matrix_is_orthonormal(body):
    R = np.array(body.truth()["cam_mat"]).reshape(3, 3)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_tof_carries_sim_time(body):
    """Rung 0's acceptance criterion (b) is `sim_time` on frames, truth AND ToF. Without it the
    only way to put an 8x8 on the simulator's clock is to assume it belongs to whichever frame was
    fetched near it -- exactly the guess this bench exists to remove.

    The world is stepped first on purpose: the fixture leaves `data.time` at 0.0, against which a
    hardcoded `0.0` would pass too. Seven steps put the clock somewhere only a live read finds."""
    for _ in range(7):
        mujoco.mj_step(body.world.model, body.world.data)
    now = float(body.world.data.time)
    assert now > 0.0, "the world did not advance; the test cannot tell a live clock from a zero"
    tof = body.depth()
    assert tof["sim_time"] == pytest.approx(now)
    assert len(tof["distance_mm"]) == len(tof["status"]) == tof["rows"] * tof["cols"]


def test_dispatch_routes_the_truth_op(body):
    got = Handler.dispatch(None, body, {"op": "truth"})
    assert "cam_pos" in got and "sim_time" in got


def test_camera_is_ahead_of_and_above_the_trunk(body):
    """Sanity on the numbers themselves. MEASURED in the STAND keyframe: the camera sits
    0.1281 m above and 0.0643 m ahead of the trunk origin. The bounds are loose around those so
    a head-pose change does not break the test, but a collapsed model (all-zero cam_xpos, which
    is what you get if nobody called mj_forward) or a sign flip does."""
    truth = body.truth()
    delta = np.array(truth["cam_pos"]) - np.array(truth["trunk"])
    assert 0.08 < delta[2] < 0.30, f"camera {delta[2]:.4f} m above the trunk"
    assert 0.02 < delta[0] < 0.15, f"camera {delta[0]:.4f} m ahead of the trunk"
