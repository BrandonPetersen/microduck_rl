"""The SLAM scenes must have metric, pinned near/far planes.

znear/zfar in MJCF are FRACTIONS of model.stat.extent, which MuJoCo auto-derives from the scene
bounding box, so any scene-geometry edit could silently change the near/far planes' metre values.
These scenes pin extent to 1.0 so the fractions mean fixed metres regardless of scene content:
ZFAR_M = 20.0 and the checker's background test both depend on the exact values holding.
"""

from pathlib import Path

import mujoco
import pytest

SCENES = Path(__file__).parent.parent / "src" / "mjlab_microduck" / "robot" / "microduck"

# The camera sits 0.2481 m off the floor in the STAND keyframe (0.2306-0.2383 m across the
# delivered session) with a 72.7 deg HFOV; 0.03 m leaves room for the beak and a wall walked into.
# 15 m covers the 11.2 m vslam room corner to corner.
MAX_NEAR_M = 0.03
MIN_FAR_M = 15.0


@pytest.mark.parametrize("scene", ["scene_vslam.xml", "scene_apartment.xml"])
def test_slam_scene_near_far_planes_are_metric_and_usable(scene):
    model = mujoco.MjModel.from_xml_path(str(SCENES / scene))
    extent = model.stat.extent
    near_m = model.vis.map.znear * extent
    far_m = model.vis.map.zfar * extent
    assert near_m <= MAX_NEAR_M, (
        f"{scene}: near plane {near_m:.4f} m exceeds the {MAX_NEAR_M} m margin -- "
        "extent may no longer be pinned"
    )
    assert far_m >= MIN_FAR_M, f"{scene}: far plane {far_m:.1f} m is too close for the room"


@pytest.mark.parametrize("scene", ["scene_vslam.xml", "scene_apartment.xml"])
def test_slam_scene_near_far_planes_are_metric_exact_contract(scene):
    """Metric values are exactly pinned for downstream consumers.

    Task 3's depth test defines ZFAR_M = 20.0 and depends on zfar being exactly that value:
    MuJoCo returns exactly zfar for pixels where no geometry was hit. A downstream mask
    triggers on this exact value, so any deviation silently breaks the contract.
    Similarly, znear must be exactly 0.02 m because extent is pinned to 1.0 and downstream
    consumers assume this exact metric value, not a value MuJoCo derives from scene geometry.
    """
    model = mujoco.MjModel.from_xml_path(str(SCENES / scene))
    extent = model.stat.extent
    near_m = model.vis.map.znear * extent
    far_m = model.vis.map.zfar * extent
    assert near_m == pytest.approx(0.02), f"{scene}: znear must be exactly 0.02 m for depth contract"
    assert far_m == pytest.approx(20.0), f"{scene}: zfar must be exactly 20.0 m for ZFAR_M constant"


@pytest.mark.parametrize("scene", ["scene_vslam.xml", "scene_apartment.xml"])
def test_slam_scene_extent_is_pinned_not_derived(scene):
    """Pinned, so a geometry edit cannot silently move the planes again."""
    model = mujoco.MjModel.from_xml_path(str(SCENES / scene))
    assert model.stat.extent == pytest.approx(1.0)
