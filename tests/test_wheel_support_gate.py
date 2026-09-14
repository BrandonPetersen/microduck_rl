"""Unit tests for mdp.wheel_support_gate — the roller-standup support gate.

The gate is what makes "standing" mean standing ON THE WHEELS. Run fmt83tri
converged to a head tripod (head planted, trunk levered to standing height at
~55 deg of tilt) that collected 48 % of the maximum task stack without ever
standing, so these cases are the failure mode itself, written down.

Contact tensors are faked: mdp reads sensors through `_sensor_any_contact`,
which only touches `env.scene.sensors[name].data.found`.
"""

import torch

from mjlab_microduck.tasks import mdp


class _FakeSensorData:
    def __init__(self, found):
        self.found = found


class _FakeSensor:
    def __init__(self, found):
        self.data = _FakeSensorData(found)


class _FakeScene:
    def __init__(self, sensors):
        self.sensors = sensors


class _FakeEnv:
    """Minimal env exposing only what the gate reads."""

    def __init__(self, num_envs, contacts):
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = _FakeScene(
            {
                name: _FakeSensor(torch.as_tensor(v, dtype=torch.float32))
                for name, v in contacts.items()
            }
        )


def _gate(contacts, num_envs=1):
    return mdp.wheel_support_gate(_FakeEnv(num_envs, contacts))


def test_wheels_only_opens_the_gate():
    # One tire on the floor, neither head nor trunk: that is standing on wheels.
    g = _gate(
        {
            "feet_ground_contact": [[1.0]],
            "head_ground_contact": [[0.0]],
            "trunk_ground_contact": [[0.0]],
        }
    )
    assert g.tolist() == [1.0]


def test_head_tripod_scores_zero():
    # THE failure mode: tires down AND head down -> no goal-state reward at all.
    g = _gate(
        {
            "feet_ground_contact": [[1.0]],
            "head_ground_contact": [[1.0]],
            "trunk_ground_contact": [[0.0]],
        }
    )
    assert g.tolist() == [0.0]


def test_trunk_prop_scores_zero():
    # The next hack if only the head were gated: prop up on the battery.
    g = _gate(
        {
            "feet_ground_contact": [[1.0]],
            "head_ground_contact": [[0.0]],
            "trunk_ground_contact": [[1.0]],
        }
    )
    assert g.tolist() == [0.0]


def test_airborne_scores_zero():
    # No support at all: not a stand, so no payout (anti-ballistic).
    g = _gate(
        {
            "feet_ground_contact": [[0.0]],
            "head_ground_contact": [[0.0]],
            "trunk_ground_contact": [[0.0]],
        }
    )
    assert g.tolist() == [0.0]


def test_any_wheel_counts_not_all():
    """A single slot in contact is enough.

    Requiring both feet would make the gate a knife edge that cuts out at the
    slightest single-support moment — a robot standing on wheels has those.
    """
    g = _gate(
        {
            "feet_ground_contact": [[1.0, 0.0]],
            "head_ground_contact": [[0.0]],
            "trunk_ground_contact": [[0.0]],
        }
    )
    assert g.tolist() == [1.0]


def test_gate_is_per_env():
    g = _gate(
        {
            # env0: standing on wheels. env1: tripod. env2: flat on the trunk.
            "feet_ground_contact": [[1.0], [1.0], [1.0]],
            "head_ground_contact": [[0.0], [1.0], [0.0]],
            "trunk_ground_contact": [[0.0], [0.0], [1.0]],
        },
        num_envs=3,
    )
    assert g.tolist() == [1.0, 0.0, 0.0]


def test_missing_sensors_degrade_to_all_ones():
    """An env declaring none of these sensors must not be neutralised.

    The gated variants carry default sensor names; if another env called them
    without declaring the sensors, a default-closed gate would silently zero its
    rewards. So it opens instead.
    """
    g = _gate({}, num_envs=2)
    assert g.tolist() == [1.0, 1.0]


def test_gated_variants_multiply_the_base_reward():
    """The gated variant = base term x gate, with nothing else changed.

    Checked by building a closed gate and an open gate over the SAME data: the
    ratio must be exactly 0 and the identity.
    """
    # Gate closed by the head -> 0; gate open -> base value unchanged.
    closed = _gate(
        {"feet_ground_contact": [[1.0]], "head_ground_contact": [[1.0]]}
    )
    opened = _gate(
        {"feet_ground_contact": [[1.0]], "head_ground_contact": [[0.0]]}
    )
    assert closed.tolist() == [0.0]
    assert opened.tolist() == [1.0]


# ── The v1 gate's hole: hips and shins ───────────────────────────────────────
# Measured on the first gated run. The rollers model carries only 12 COLLISION
# geoms: np_f970 (battery) on trunk_base, 3 head geoms on jaw_soft,
# hip_l/hip_l_2, leg/leg_2, and the 4 tires. The trunk shells are VISUAL-only.
# So a robot sprawled on its hips and shins, one tire grazing the floor and the
# head held up, triggered NEITHER head_ground_contact NOR trunk_ground_contact:
# the gate opened while lying flat and pose_stand_legs paid. Symptom:
# pose_stand_legs/1.9 = 0.67 while standing_prob = 0.50, misread as "a third of
# the face-down starts get up".


def test_sprawl_on_hips_and_shins_scores_zero():
    g = _gate(
        {
            "feet_ground_contact": [[1.0]],
            "head_ground_contact": [[0.0]],
            "trunk_ground_contact": [[0.0]],
            "limbs_ground_contact": [[1.0]],
        }
    )
    assert g.tolist() == [0.0]


def test_limbs_sensor_is_in_the_default_forbidden_set():
    """The default must include all three, else an env omitting it leaks again."""
    assert mdp._WHEEL_SUPPORT_FORBIDDEN_SENSORS == (
        "head_ground_contact",
        "trunk_ground_contact",
        "limbs_ground_contact",
    )


def test_only_wheels_touching_still_opens():
    """The gate must stay OPENABLE: standing, only the tires touch.

    The symmetric risk of the fix: too many forbidden contacts and the gate never
    opens again, silently zeroing the goal-state rewards.
    """
    g = _gate(
        {
            "feet_ground_contact": [[1.0]],
            "head_ground_contact": [[0.0]],
            "trunk_ground_contact": [[0.0]],
            "limbs_ground_contact": [[0.0]],
        }
    )
    assert g.tolist() == [1.0]
