"""The SLAM scenes must not clip what the duck's camera can actually see.

znear/zfar in MJCF are FRACTIONS of model.stat.extent, which MuJoCo auto-derives from the scene
bounding box. A big room therefore pushes the near plane out: scene_vslam.xml measured 0.2238 m,
which clips the nearest floor in frame and any wall the duck walks up to -- in RGB as well as
depth. These scenes pin extent so the fractions mean fixed metres.
"""

from pathlib import Path

import mujoco
import pytest

SCENES = Path(__file__).parent.parent / "src" / "mjlab_microduck" / "robot" / "microduck"

# The camera is 0.37 m off the floor with a 72.7 deg HFOV; 0.03 m leaves room for the beak and a
# wall walked into. 15 m covers the 11.2 m vslam room corner to corner.
MAX_NEAR_M = 0.03
MIN_FAR_M = 15.0


@pytest.mark.parametrize("scene", ["scene_vslam.xml", "scene_apartment.xml"])
def test_slam_scene_near_far_planes_are_metric_and_usable(scene):
    model = mujoco.MjModel.from_xml_path(str(SCENES / scene))
    extent = model.stat.extent
    near_m = model.vis.map.znear * extent
    far_m = model.vis.map.zfar * extent
    assert near_m <= MAX_NEAR_M, f"{scene}: near plane {near_m:.4f} m clips the duck's own view"
    assert far_m >= MIN_FAR_M, f"{scene}: far plane {far_m:.1f} m is too close for the room"


@pytest.mark.parametrize("scene", ["scene_vslam.xml", "scene_apartment.xml"])
def test_slam_scene_extent_is_pinned_not_derived(scene):
    """Pinned, so a geometry edit cannot silently move the planes again."""
    model = mujoco.MjModel.from_xml_path(str(SCENES / scene))
    assert model.stat.extent == pytest.approx(1.0)
