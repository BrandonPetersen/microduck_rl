"""The head camera's frame, pinned.

Two things must hold and neither was tested before:

1. The camera points FORWARD (trunk +x) straight out of the MJCF, with no Python fix-up. The fix
   used to live in Camera.__init__, so anything reading data.cam_xmat without building a Camera
   (the truth op, a debug script) saw a camera facing backwards.

2. The rendered frame is rolled +90 deg about the optical axis relative to the FK-published camera
   frame (robot.state.frames.camera). This is DELIBERATE: the real head camera is mounted a quarter
   turn off, media.video.rotate=90 announces it, and duckslam's upright() undoes it. The twin
   reproduces the real mount. This test exists so nobody "fixes" it.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

SCENES = Path(__file__).parent.parent / "src" / "mjlab_microduck" / "robot" / "microduck"

# kinematics/src/head.rs SITE_TO_CV2: the FK camera frame has optical +z along trunk +x,
# image +x along trunk -y, image +y along trunk -z.
R_FK_IN_TRUNK = np.array([
    [0.0, 0.0, 1.0],    # trunk x <- (right, down, forward)
    [-1.0, 0.0, 0.0],   # trunk y
    [0.0, -1.0, 0.0],   # trunk z
])


def _model_data(scene="scene_vslam.xml"):
    model = mujoco.MjModel.from_xml_path(str(SCENES / scene))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data


def test_head_camera_faces_forward_without_any_python_fixup():
    model, data = _model_data()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    # MuJoCo cameras look along their own -z; cam_xmat columns are the camera axes in world.
    view_dir_world = -data.cam_xmat[cam].reshape(3, 3)[:, 2]
    trunk_forward_world = data.xmat[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    ].reshape(3, 3)[:, 0]
    assert float(view_dir_world @ trunk_forward_world) > 0.9, (
        f"camera looks {view_dir_world} where the trunk faces {trunk_forward_world}"
    )


def test_rendered_frame_is_rolled_90_deg_from_the_fk_frame():
    """R_FK^T @ R_rendered == Rz(+90 deg). Intentional -- do not 'fix' this."""
    model, data = _model_data()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    R_trunk_world = data.xmat[trunk].reshape(3, 3)
    # MuJoCo camera axes are (right, up, backward); OpenCV optical is (right, down, forward).
    cv_from_mj = np.diag([1.0, -1.0, -1.0])
    R_rendered_in_trunk = R_trunk_world.T @ data.cam_xmat[cam].reshape(3, 3) @ cv_from_mj
    roll = R_FK_IN_TRUNK.T @ R_rendered_in_trunk
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Rz(+90 deg)
    np.testing.assert_allclose(roll, expected, atol=1e-6)


def test_camera_construction_does_not_mutate_the_model():
    from mjlab_microduck.sim.camera import Camera

    model, _ = _model_data()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    before = model.cam_quat[cam].copy()
    Camera(model, "head_camera")
    np.testing.assert_array_equal(model.cam_quat[cam], before)
