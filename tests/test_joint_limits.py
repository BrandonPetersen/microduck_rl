"""The knee and ankle ranges are MEASURED, and must not silently revert.

onshape-to-robot writes a joint range only where the Onshape mate declares a
limit. hip_pitch, knee and ankle had none, so every model variant shipped with
exactly +/-pi/2 on all six -- a placeholder that reads as a real number.

That cost a servo. On 2026-09-24 the HopFree hop policy commanded the ankle to
+/-102.8 deg against a real stop at ~76 deg, with a position controller at
gain 400 behind it. The servo drove into the stop, stalled, and latched
Hardware Error bit 5 (overload), after which it refused torque and the robot
would not stand. Nothing catches this below the MJCF: the servos' own Position
Limit registers read [-180.0, +179.9] and duck-control's safety.rs clamps only
to the ACTUATOR's travel, saying in a comment that "the alpha robot's real
joint limits live in the MJCF".

Measured by hand with torque off (scripts/robot/measure_travel.sh), both legs,
several sweeps. The knee cross-validates between sides to 1.0 deg; the ankle
was measured on the left four times and on the right twice, agreeing to 2.4.

A RE-EXPORT FROM ONSHAPE WILL REVERT THESE unless the mates get real limits.
That is what this test is for.
"""

import glob
import math
import os
import re

import pytest

MODELS = os.path.join(
    os.path.dirname(__file__), "..", "src", "mjlab_microduck", "robot", "microduck", "robot_*.xml"
)
# radians; see the module docstring for provenance
EXPECTED = {
    "left_ankle": (-1.308997, +1.308997),
    "right_ankle": (-1.308997, +1.308997),
    "left_knee": (-1.989675, +1.396263),
    "right_knee": (-1.396263, +1.989675),
}


def _range(text: str, joint: str):
    m = re.search(rf'name="{joint}" type="hinge" range="([-0-9.]+) ([-0-9.]+)"', text)
    return (float(m.group(1)), float(m.group(2))) if m else None


@pytest.mark.parametrize("path", sorted(glob.glob(MODELS)))
def test_knee_and_ankle_carry_the_measured_limits(path):
    text = open(path).read()
    for joint, (lo, hi) in EXPECTED.items():
        got = _range(text, joint)
        if got is None:
            continue           # not every variant has every joint
        assert got == pytest.approx((lo, hi), abs=1e-5), (
            f"{os.path.basename(path)} {joint} is {got}, expected {(lo, hi)}. "
            "If this came from a fresh onshape-to-robot export, the Onshape mate "
            "still has no limit and the placeholder has come back."
        )


@pytest.mark.parametrize("path", sorted(glob.glob(MODELS)))
def test_no_knee_or_ankle_is_still_the_pi_over_2_placeholder(path):
    text = open(path).read()
    for joint in EXPECTED:
        got = _range(text, joint)
        if got is None:
            continue
        for bound in got:
            assert abs(abs(bound) - math.pi / 2) > 1e-6, (
                f"{os.path.basename(path)} {joint} is back to +/-pi/2, the export default"
            )


def test_the_home_pose_sits_inside_the_measured_limits():
    """A limit tighter than the home pose would make the rest pose unreachable,
    which is the one way this change could brick every policy at once."""
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME

    import re as _re

    # HOME_FRAME keys are REGEX patterns (r".*left_ankle.*"), not literal names,
    # so a dict lookup silently finds nothing and the test would pass vacuously.
    pose = HOME_FRAME.joint_pos
    checked = 0
    for joint, (lo, hi) in EXPECTED.items():
        for pattern, home in pose.items():
            if _re.fullmatch(pattern, joint):
                assert lo < home < hi, (
                    f"{joint} home {home} is outside the measured range [{lo}, {hi}] -- "
                    "the rest pose would be unreachable"
                )
                checked += 1
                break
    assert checked == len(EXPECTED), f"only matched {checked} of {len(EXPECTED)} joints"
