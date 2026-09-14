import pytest

from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
    EPISODE_LENGTH_S,
    NUM_STEPS_PER_ENV,
    make_microduck_roller_standup_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)

# SKATING rewards: none of them may survive in a standup env.
SKATING_REWARDS = (
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
)


def test_env_builds_train_and_play():
    assert make_microduck_roller_standup_env_cfg() is not None
    assert make_microduck_roller_standup_env_cfg(play=True) is not None


def test_episode_is_short():
    # Short episode: rise then stabilise, same as standup (6 s).
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 6.0


def test_no_skating_rewards_survive():
    cfg = make_microduck_roller_standup_env_cfg()
    for name in SKATING_REWARDS:
        assert name not in cfg.rewards, f"skating reward survived: {name}"


def test_smoothness_regularisers_kept():
    # Kept from the roller inheritance: the standup needs sim2real smoothness,
    # but body_ang_vel must stay LIGHT (standup documents that -0.15 froze it).
    cfg = make_microduck_roller_standup_env_cfg()
    for name in (
        "action_over_limit",
        "self_collisions",
        "body_ang_vel",
        "angular_momentum",
        "action_rate_l2",
        "neck_joint_pos_l2",
        "joint_torques_l2",
    ):
        assert name in cfg.rewards, f"regulariser lost: {name}"
    assert cfg.rewards["body_ang_vel"].weight == -0.05


def test_neck_action_rate_is_dropped_but_neck_position_is_kept():
    """The two neck terms pull in OPPOSITE directions — only one must go.

    neck_action_rate_l2 taxes head MOTION: it double-taxes the 4 joints already
    covered by action_rate_l2 (effective weight 0.6 against 0.1 per leg joint at
    stage 0), measured at -1.359/step on the smoke, the largest term in the whole
    reward. That is a tax on attempts during discovery, and the reference recipe
    drops it (microduck_standup_env_cfg.py:485).

    neck_joint_pos_l2 taxes the head being FAR FROM NEUTRAL: that is what fights
    the head tripod. It stays.

    The trap this test locks in: "remove the neck penalty" without saying which.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert "neck_action_rate_l2" not in cfg.rewards
    assert "neck_joint_pos_l2" in cfg.rewards
    assert cfg.rewards["neck_joint_pos_l2"].weight == -0.5


def test_action_over_limit_is_kept():
    """A sim2real guard, not a tax on motion.

    It penalises commands outside ctrlrange — a specific pathology — and it is the
    roller family's policy-side protection. Dropping it alongside
    neck_action_rate_l2 would be a transfer risk with no measured benefit.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.rewards["action_over_limit"].weight == -0.5


def test_face_up_spawns_get_roll_noise():
    """Built-in reverse curriculum on the hard case.

    Without this parameter (default 0.0), ALL back starts are flat — the case the
    walker documents as having a flat reward landscape until the roll completes,
    hence "seed-lucky" success (1 out of 4). The roll noise makes a fraction of
    episodes begin part-way through the roll, which finally gives the end of the
    gesture some on-policy data.
    """
    import math

    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.events["set_ground_state"].params["face_up_roll_max"] == pytest.approx(
        math.radians(90)
    )


def test_backlash_variant_keeps_the_spawn_roll_noise():
    import math

    bl = _load("Mjlab-RollerStandUp-Flat-Backlash-MicroDuck")
    assert bl.events["set_ground_state"].params["face_up_roll_max"] == pytest.approx(
        math.radians(90)
    )
    assert "neck_action_rate_l2" not in bl.rewards


def test_twist_command_is_neutralised():
    # Nothing is steered: the policy deploys in --standing, where the runtime
    # leaves the twist slot at zero (see infer_policy.py:239).
    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.rel_standing_envs == 0.0


def test_twist_command_is_not_heading_relative():
    # The roller env installs a RelativeHeadingVelocityCommandCfg (cmd[2] =
    # heading error, computed internally). Here cmd[2] must be a true noisy zero.
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert not isinstance(cmd, microduck_mdp.RelativeHeadingVelocityCommandCfg)


def test_obs_nan_policy_sanitize():
    # A rare contact diverges the free joint into NaN: we sanitise the obs rather
    # than kill training (same choice as roller_slope).
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"


def test_obs_parity_with_roller_env():
    # 61D parity is mandatory: otherwise the ONNX will not load into a runtime slot.
    standup = make_microduck_roller_standup_env_cfg()
    roller = make_microduck_velocity_rollers_env_cfg()
    for grp in ("actor", "critic"):
        assert list(standup.observations[grp].terms.keys()) == list(
            roller.observations[grp].terms.keys()
        ), f"observation layout diverged on group {grp}"


def test_terrain_is_plain_plane():
    # Inherited from the roller env: flat ground, no generator. No rough variant
    # for this v1.
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401  (importing triggers registration)

    assert "Mjlab-RollerStandUp-Flat-MicroDuck" in list_tasks()


def test_joint_indices_are_in_the_canonical_servo_space():
    """Lock: the indices target the SERVO-ONLY view, not the full array.

    pose_target_match / pose_l1_penalty / standing_composite_score index through
    mdp._servo_joint_pos, which selects `^(?!passive_).*` — so the 14 servos,
    excluding ALL passive joints (wheels AND backlash hinges). The indices must
    therefore be written in the canonical 14-joint layout, the same as the
    walker's, and NOT in the rollers entity's 18-joint array.

    History: this file used [0-4, 11-15] (the real positions in the rollers
    model's full array). After mdp.py migrated to _servo_joint_pos, indices 14 and
    15 fell outside a 14-column tensor → "index out of bounds" on GPU, env
    untrainable. Config tests could not catch it: they never call the rewards.

    We check on BOTH roller models — plain and backlash — because the invariant
    "the servo-only view is identical" is exactly what makes a backlash variant of
    this env safe to register.
    """
    import mujoco

    from mjlab_microduck.robot.microduck_constants import (
        get_rollers_backlash_spec,
        get_walk_rollers_spec,
    )
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
        _LEG_JOINTS,
        _NECK_JOINTS,
    )

    expected_legs = [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    expected_neck = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]

    for label, spec_fn in (
        ("rollers", get_walk_rollers_spec),
        ("rollers_backlash", get_rollers_backlash_spec),
    ):
        model = spec_fn().compile()
        servo = []
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name.startswith("passive_"):
                servo.append(name)

        assert len(servo) == 14, f"{label}: {len(servo)} servos instead of 14"
        assert max(_LEG_JOINTS + _NECK_JOINTS) < len(servo), (
            f"{label}: index out of bounds of the servo-only view"
        )
        assert [servo[i] for i in _LEG_JOINTS] == expected_legs, label
        assert [servo[i] for i in _NECK_JOINTS] == expected_neck, label
        # The two lists cover exactly the 14 servos, with no overlap.
        assert len(set(_LEG_JOINTS) | set(_NECK_JOINTS)) == 14, label


def test_recovery_rewards_present_with_expected_weights():
    """Weights aligned on the evolved standup recipe (task block / 4).

    The whole task block was divided by 4 while the dampers kept their values:
    that is the task/damper ratio correction, measured at ~35:1 (task ≈ +41.6
    against ≈ -1.2 of dampers), which left no reason to be gentle. Dividing the
    task rather than raising the dampers avoids turning one into a motion-blocker.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    expected = {
        "pose_stand_legs":      2.0,
        "pose_stand_l1":        1.25,
        "height_stand":         1.0,
        "height_stand_sharp":   1.0,
        "height_stand_l1":      7.5,
        "com_upward_velocity":  0.75,
        # POSITIVE: trunk_vertical_accel_penalty already returns -|a_z|.
        "gentle_rise":          0.005,
        "upright_linear":       1.5,
        "upright_sharp":        1.5,
        "standing_composite":   3.75,
        # Potential-based Δz, the gradient off the floor. See
        # test_height_progress_weight_cancels_the_action_tax for the derivation.
        "height_progress":      200.0,
        # 0 at start: introduced late by torque_rate_weight (see
        # test_anti_violence_terms_are_introduced_late).
        "joint_torque_rate_l2": 0.0,
        "arrival_damping":      0.0,
    }
    for name, weight in expected.items():
        assert name in cfg.rewards, f"missing reward: {name}"
        assert cfg.rewards[name].weight == weight, f"unexpected weight on {name}"


def test_recovery_rewards_use_roller_heights_not_walker_heights():
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
        ROLLER_PRONE_Z,
        ROLLER_STAND_Z,
    )

    cfg = make_microduck_roller_standup_env_cfg()
    assert ROLLER_STAND_Z == 0.138  # NOT the 0.115 of the model without wheels
    for name in ("height_stand", "height_stand_sharp", "height_stand_l1"):
        assert cfg.rewards[name].params["target_height"] == ROLLER_STAND_Z
    assert cfg.rewards["standing_composite"].params["target_height"] == ROLLER_STAND_Z
    # com_upward_velocity cuts off just ABOVE the target (10 mm of margin),
    # otherwise the policy parks at the cutoff altitude without finishing the rise.
    assert cfg.rewards["com_upward_velocity"].params["max_height"] == ROLLER_STAND_Z + 0.010
    # upright_sharp is gated between the ground rest height and the stand.
    assert cfg.rewards["upright_sharp"].params["height_low"] == ROLLER_PRONE_Z
    assert cfg.rewards["upright_sharp"].params["height_high"] == ROLLER_STAND_Z


def test_pose_rewards_target_legs_only_at_roller_indices():
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import _LEG_JOINTS

    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("pose_stand_legs", "pose_stand_l1", "standing_composite"):
        assert cfg.rewards[name].params["joint_indices"] == _LEG_JOINTS
        # target_overrides=None → the target is HOME (default_joint_pos).
        assert cfg.rewards[name].params["target_overrides"] is None


def test_trunk_asset_cfgs_are_distinct_objects():
    """mjlab resolves and MUTATES SceneEntityCfg in place: an object shared across
    several terms causes stale indices. Each term must own its own.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    names = (
        "height_stand", "height_stand_sharp", "height_stand_l1",
        "com_upward_velocity", "gentle_rise", "upright_linear",
        "upright_sharp", "standing_composite",
    )
    seen = [id(cfg.rewards[n].params["asset_cfg"]) for n in names]
    assert len(set(seen)) == len(seen), "asset_cfg shared between several terms"


def test_starts_from_ground_states():
    # Face-down + face-up + standing. No "sitting" bucket: in standup it existed
    # only for the hand-off from the sit policy, which has no roller equivalent —
    # and its sitting_joint_overrides are indices of the model WITHOUT wheels.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "set_ground_state" in cfg.events
    params = cfg.events["set_ground_state"].params
    assert params["sitting_prob"] == 0.0
    assert params["sitting_joint_overrides"] is None
    assert params["face_down_prob"] > 0.0
    assert params["standing_prob"] > 0.0
    # face_up (the back) starts at 0: introduced late by the curriculum.
    assert params["face_up_prob"] == 0.0


def test_ground_state_heights_are_roller_specific():
    cfg = make_microduck_roller_standup_env_cfg()
    params = cfg.events["set_ground_state"].params
    # Face-down and face-up share a single z range, but their contacts differ:
    # the belly only clears the floor from 0.0752, the back rests at 0.0475.
    # prone_z_min = 0.076 to eliminate any interpenetration on the face-down side.
    assert (params["prone_z_min"], params["prone_z_max"]) == (0.076, 0.09)
    # Below 0.0752 (measured contact, HOME pose), a face-down start begins INSIDE
    # the floor — a contact pushout the policy would pay for through gentle_rise /
    # joint_torque_rate_l2. prone_z_min must stay above it.
    assert params["prone_z_min"] >= 0.0752
    # Standing: ROLLER height (+23 mm vs the model without wheels, at 0.11–0.12).
    assert params["standing_z_min"] == 0.134
    assert params["standing_z_max"] == 0.144
    assert params["standing_z_min"] < 0.138 < params["standing_z_max"]


def test_ground_state_event_runs_after_base_reset():
    # set_ground_state overwrites the pose set by reset_base / reset_robot_joints:
    # event order follows insertion order, so it must come AFTER them.
    cfg = make_microduck_roller_standup_env_cfg()
    order = list(cfg.events.keys())
    assert order.index("set_ground_state") > order.index("reset_base")
    assert order.index("set_ground_state") > order.index("reset_robot_joints")


def test_no_fall_termination():
    # The robot STARTS fallen: a tilt termination would kill the episode on the
    # first step. nan_state (inherited) does stay.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_ground_state_curriculum_ramps_easy_to_hard():
    cfg = make_microduck_roller_standup_env_cfg()
    assert "ground_state_mix" in cfg.curriculum
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert cfg.curriculum["ground_state_mix"].params["event_name"] == "set_ground_state"
    # Steps are increasing and start at 0.
    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)
    # The back (face_up) is introduced late then grows monotonically.
    face_up = [s["params"]["face_up_prob"] for s in stages]
    assert face_up[0] == 0.0
    assert face_up == sorted(face_up)
    assert face_up[-1] >= 0.35
    # Every stage is a valid distribution, and "already standing" never
    # disappears (otherwise the policy rises then falls, never learning to hold).
    for stage in stages:
        p = stage["params"]
        total = (
            p["standing_prob"] + p["sitting_prob"]
            + p["face_down_prob"] + p["face_up_prob"]
        )
        assert abs(total - 1.0) < 1e-9
        assert p["sitting_prob"] == 0.0
        assert p["standing_prob"] > 0.0


def test_wheel_friction_curriculum_is_decreasing():
    """The env's new piece: BRAKED wheels → FREE wheels.

    The wheels roll, so there is no longitudinal grip to push against the floor.
    We bootstrap with nearly-locked bearings (the standup then works like it would
    on feet) and ramp toward the real value. The roller env, by contrast, RAISES
    this friction (0 → 0.0015): the direction really is inverted here.

    NOTE: run fmt83tri measured this curriculum to be a non-event — no drop in
    standing_composite at any of its three stages. The direction assertion stays
    as a lock on intent, but the curriculum itself is a compression candidate.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    stages = cfg.curriculum["wheel_friction"].params["ranges_stages"]
    assert cfg.curriculum["wheel_friction"].params["event_name"] == "randomize_wheel_friction"

    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)

    lows = [s["ranges"][0] for s in stages]
    assert lows == sorted(lows, reverse=True), "friction must DECREASE"
    assert lows[0] >= 0.02, "start firmly braked to bootstrap the gesture"
    # Ends on the real rolling value (the roller env's).
    assert stages[-1]["ranges"] == (0.0015, 0.0015)
    for stage in stages:
        assert stage["ranges"][0] == stage["ranges"][1]


def test_wheel_friction_event_default_matches_stage_zero():
    # The curriculum manager runs BEFORE the reset events on every reset
    # (including the very first), and wheel_friction_curriculum itself defaults to
    # stage 0: the event's default value is therefore never read in practice. We
    # only check it stays consistent with the curriculum's stage 0 — defensive
    # redundancy, useful if the curriculum ever disappears and leaves the event.
    cfg = make_microduck_roller_standup_env_cfg()
    stage0 = cfg.curriculum["wheel_friction"].params["ranges_stages"][0]["ranges"]
    assert cfg.events["randomize_wheel_friction"].params["ranges"] == stage0


def test_push_curriculum_ramps_from_zero():
    # Inherited pushes (±0.2 m/s), but ramped: a shove from step 0 disturbs the
    # standup bootstrap.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "push_robot" in cfg.events
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert cfg.curriculum["push_magnitude"].params["event_name"] == "push_robot"
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)
    assert stages[-1]["velocity_range"]["x"] == (-0.2, 0.2)
    highs = [s["velocity_range"]["x"][1] for s in stages]
    assert highs == sorted(highs), "the push must GROW"


def test_inherited_dr_curricula_survive():
    # The DR inherited from the roller env must not have been lost along the way.
    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("com_range", "head_com_range"):
        assert name in cfg.curriculum, f"DR curriculum lost: {name}"
    for name in (
        "randomize_com",
        "randomize_head_com",
        "randomize_armature",
        "randomize_joint_friction",
        "randomize_mass_inertia",
        "randomize_wheel_friction",
        "encoder_bias",
    ):
        assert name in cfg.events, f"DR event lost: {name}"


# ── Play override: forcing back starts ────────────────────────────────────────
# Without the override, a play NEVER shows a back start: the play env is rebuilt
# from scratch, so common_step_counter restarts at 0 and the curriculum applies
# its stage 0, where face_up_prob = 0. Yet that is precisely the hardest case, the
# one worth inspecting by eye. STANDUP_PLAY_FACE_UP forces the mix, following the
# SLOPE_PLAY_DIFFICULTY pattern in roller_slope.


def test_play_face_up_override_forces_back_starts(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "1.0")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    params = cfg.events["set_ground_state"].params
    assert params["face_up_prob"] == 1.0
    assert params["face_down_prob"] == 0.0
    assert params["standing_prob"] == 0.0
    # Without this, the curriculum would rewrite the probabilities on the very
    # first reset (event_param_curriculum runs BEFORE the reset events).
    assert "ground_state_mix" not in cfg.curriculum


def test_play_face_up_override_splits_remainder_like_final_stage(monkeypatch):
    # 0.4 must reproduce the LAST curriculum stage (0.40 face-down / 0.20
    # standing / 0.40 back): the remainder is split in that stage's 2:1 ratio.
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "0.4")
    params = make_microduck_roller_standup_env_cfg(play=True).events["set_ground_state"].params
    assert params["face_up_prob"] == pytest.approx(0.40)
    assert params["face_down_prob"] == pytest.approx(0.40)
    assert params["standing_prob"] == pytest.approx(0.20)
    total = params["face_up_prob"] + params["face_down_prob"] + params["standing_prob"]
    assert total == pytest.approx(1.0)


def test_play_face_up_override_is_clamped(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "3.0")
    params = make_microduck_roller_standup_env_cfg(play=True).events["set_ground_state"].params
    assert params["face_up_prob"] == 1.0


def test_play_face_up_override_ignored_during_training(monkeypatch):
    # Guard rail: the variable must NEVER touch training, otherwise we would break
    # the easy->hard curriculum without noticing.
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "1.0")
    cfg = make_microduck_roller_standup_env_cfg(play=False)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_without_override_keeps_curriculum_mix(monkeypatch):
    # Default behaviour unchanged: stage 0, no back start.
    monkeypatch.delenv("STANDUP_PLAY_FACE_UP", raising=False)
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_face_up_override_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "nonsense")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_face_up_override_none_keyword_disables(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "none")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


# ── Anti-violence: fixes after testing on the robot ───────────────────────────
# Symptoms observed (checkpoint 4000+, IN SIM TOO so not a sim2real issue): very
# abrupt motion, the head banging the floor, back recovery failing on the real
# robot. Diagnosis measured in wandb (run vweolw91, iteration 7500).


def test_already_negative_penalties_use_positive_weights():
    """Lock on the bug class that made the policy violent.

    mdp.py mixes TWO sign conventions: some penalty functions return a positive
    magnitude (to be multiplied by a negative weight), others already return a
    negative value (to be multiplied by a POSITIVE weight).
    trunk_vertical_accel_penalty returns -|a_z|: with the -0.02 weight inherited
    from standup, the double negative REWARDED vertical acceleration — measured at
    Episode_Reward/gentle_rise = +0.0118, the only penalty term logging positive.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    # These three terms call functions that already return negatives
    # (height_l1_penalty, pose_l1_penalty, trunk_vertical_accel_penalty).
    for name in ("height_stand_l1", "pose_stand_l1", "gentle_rise"):
        assert cfg.rewards[name].weight > 0, (
            f"{name} calls a function that already returns a negative: "
            f"a negative weight would turn it into a reward"
        )
    # And these return a positive magnitude → negative weight (or 0 while a late
    # curriculum introduces them: joint_torque_rate_l2, arrival_damping).
    for name in ("joint_torques_l2", "action_rate_l2"):
        assert cfg.rewards[name].weight < 0, f"{name} expects a negative weight"
    for name in ("joint_torque_rate_l2", "arrival_damping"):
        assert cfg.rewards[name].weight <= 0, f"{name} must never be positive"


def test_no_ungated_head_impact_penalty():
    """NO ungated head-impact penalty — it froze the policy.

    Tried at -1.0 (velstand's values): the policy converged to lying down, inert.
    Measured on run d8rnko6p: head_impact_penalty -1.01/step, the largest negative
    term, while standing_composite collapsed from +14.3 to +3.3.

    The reasoning error was believing that a "targeted" penalty does not restrict
    motion. False here: to get up from its back, this robot PIVOTS on its head and
    shoulders. The head is the rollover's support point, not collateral damage —
    penalising it means penalising the only available mechanism.

    If the slam comes back once the gentle_rise sign is fixed, the reintroduction
    must be a HEIGHT-GATED penalty (the way upright_sharp is) that spares the
    ground rollover phase. Not this one.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert "head_impact_penalty" not in cfg.rewards
    assert "head_impact_contact" not in [s.name for s in cfg.scene.sensors]


def test_inherited_sensors_intact():
    # The sensors inherited from the roller env are used by kept rewards
    # (self_collisions) and by the observations.
    cfg = make_microduck_roller_standup_env_cfg()
    names = [s.name for s in cfg.scene.sensors]
    assert "feet_ground_contact" in names
    assert "self_collision" in names


def test_height_l1_stays_the_dominant_task_term():
    """The freeze comes from a lazy optimum: lying down, legs at HOME, pays.

    pose_stand_legs stayed at +7.72 out of 8 while the robot was lying flat — the
    legs sit at HOME in a lying pose, so the pose reward is collected almost for
    free. height_stand_l1 is the term that counterbalances that by making "stay on
    the ground" net NEGATIVE (it is -|z - target|, so -0.063 × weight when
    face-down). It must remain the heaviest of the task block.

    Scale-invariant assertion: the whole block has already been divided by 4 once,
    so we check the RATIO rather than an absolute value.

    height_progress is deliberately NOT in this comparison: its weight multiplies
    a per-step Δz in metres (~0.001), not a level in [0, 1], so 200 there is not
    comparable to 7.5 here. Do not "fix" the two into agreement.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    task_terms = (
        "pose_stand_legs", "pose_stand_l1", "height_stand", "height_stand_sharp",
        "height_stand_l1", "upright_linear", "upright_sharp", "standing_composite",
    )
    weights = {n: abs(cfg.rewards[n].weight) for n in task_terms}
    assert weights["height_stand_l1"] == max(weights.values()), (
        f"height_stand_l1 must dominate the task block, got {weights}"
    )
    # And it must stay clearly above the pose term, which is the "free while lying
    # down" term it counterbalances.
    assert weights["height_stand_l1"] >= 3.0 * weights["pose_stand_legs"]
    assert cfg.rewards["com_upward_velocity"].weight > 0.0


def test_anti_violence_terms_are_introduced_late():
    """The freeze came from TIMING, not magnitude.

    Lesson established across two broken standup runs, quoted in its comments:
    "the same weights active from step 0 prevent the flips from ever being
    DISCOVERED (attempt-tax on exploration)". ground_state_mix finishes ramping
    the hard poses at iteration 2500; the anti-violence penalties therefore only
    enter at 3000, when the skills exist and ground resets keep exercising them.

    That is exactly what froze this env: head_impact_penalty (-1.0) and
    joint_torque_rate_l2 (-2.0) were active from step 0.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    for cur_name, reward_name in (
        ("arrival_damping_weight", "arrival_damping"),
        ("torque_rate_weight", "joint_torque_rate_l2"),
    ):
        assert cur_name in cfg.curriculum, f"missing curriculum: {cur_name}"
        stages = cfg.curriculum[cur_name].params["weight_stages"]
        assert cfg.curriculum[cur_name].params["reward_name"] == reward_name
        assert stages[0]["step"] == 0 and stages[0]["weight"] == 0.0, (
            f"{reward_name} must start at 0"
        )
        # Nothing before 3000 iters: ground_state_mix finishes ramping at 2500.
        first_active = min(s["step"] for s in stages if s["weight"] != 0.0)
        assert first_active >= 3000 * NUM_STEPS_PER_ENV, (
            f"{reward_name} introduced too early (tax on exploration)"
        )


def test_arrival_damping_gates_are_scaled_to_roller_height():
    """The gate must be relative to the ROLLER standing height, not the walker's.

    standup uses 0.09/0.11 for STAND_Z=0.115, i.e. -25 mm and -5 mm below the
    stand. Copied verbatim onto the roller (0.138), those bounds would open the
    gate while the robot is still ~3 cm below its standing height, i.e. mid-rise —
    exactly what the gate is supposed to spare.
    """
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import ROLLER_STAND_Z

    params = make_microduck_roller_standup_env_cfg().rewards["arrival_damping"].params
    assert params["height_low"] == pytest.approx(ROLLER_STAND_Z - 0.025)
    assert params["height_high"] == pytest.approx(ROLLER_STAND_Z - 0.005)
    assert params["height_low"] < params["height_high"] < ROLLER_STAND_Z
    # The tilt gate is essential: without it, the final straightening of a folded
    # rise is itself a large trunk rotation, and taxing it raises a wall just
    # before arrival (standup's lesson).
    assert params["tilt_full_deg"] == 20.0
    assert params["tilt_zero_deg"] == 45.0


def test_motion_blockers_stay_light():
    """body_ang_vel and action_rate remain motion-blockers.

    standup documents that at -0.15 and -1.2 respectively they FROZE back
    recovery. Its action_rate ramp is also much gentler than before: -0.1 at the
    start, -1.0 only at 1500 iters.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.rewards["body_ang_vel"].weight == -0.05
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert weights[0] == -0.1, "a gentle start is required"
    assert min(weights) >= -1.0, "beyond -1.0 froze the standup (standup)"
    assert weights == sorted(weights, reverse=True), "the ramp must tighten"
    # -1.0 not before 1500 iters (against 500 in the previous version).
    assert min(s["step"] for s in stages if s["weight"] == -1.0) >= 1500 * NUM_STEPS_PER_ENV


# ── Backlash variant ──────────────────────────────────────────────────────────
# ±1° of gear backlash in series per servo, with the firmware PD closing on the
# encoder THROUGH the play — like the real servo, whose encoder sits on the
# gearbox output. Every other family (Rollers, RollerCrouch, RollerSlope, StandUp,
# Sit, SitStand…) has its variant; this one was missing, because "register
# backlash for all envs" happened while this env lived on another branch.
#
# It is only SAFE since the move to servo-only indices: on the backlash model the
# joint array grows to 32 entries (18 passive), so the old [0-4, 11-15] indices
# would have silently rewarded wheels and backlash hinges.


def _load(task_id):
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_microduck.tasks  # noqa: F401  (importing triggers registration)

    return load_env_cfg(task_id)


def test_backlash_variant_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    assert "Mjlab-RollerStandUp-Flat-Backlash-MicroDuck" in list_tasks()


def test_backlash_variant_keeps_the_recovery_recipe():
    # Backlash must change ONLY the robot model and how joints are read. Rewards,
    # weights, ground reset and curricula are the base task's.
    bl = _load("Mjlab-RollerStandUp-Flat-Backlash-MicroDuck")
    base = _load("Mjlab-RollerStandUp-Flat-MicroDuck")

    assert set(bl.rewards.keys()) == set(base.rewards.keys())
    for name in bl.rewards:
        assert bl.rewards[name].weight == base.rewards[name].weight, name

    assert "set_ground_state" in bl.events
    assert bl.events["set_ground_state"].params["prone_z_min"] == 0.076
    for cur in ("ground_state_mix", "wheel_friction", "arrival_damping_weight",
                "torque_rate_weight", "push_magnitude"):
        assert cur in bl.curriculum, f"curriculum lost in the variant: {cur}"
    assert "fell_over" not in bl.terminations


def test_backlash_variant_reads_joints_through_the_backlash():
    from mjlab_microduck.tasks import mdp as microduck_mdp

    bl = _load("Mjlab-RollerStandUp-Flat-Backlash-MicroDuck")
    for grp in ("actor", "critic"):
        terms = bl.observations[grp].terms
        assert terms["joint_pos"].func is microduck_mdp.joint_pos_rel_backlash
        assert terms["joint_vel"].func is microduck_mdp.joint_vel_rel_backlash
    # nan_policy preserved (make_backlash_variant does not touch it)
    assert bl.observations["actor"].nan_policy == "sanitize"


def test_backlash_variant_wheel_friction_targets_only_wheels():
    """Rolling-friction DR must NOT touch the backlash hinges.

    On the backlash model, `^passive_.*` would also match
    `passive_left_hip_yaw_backlash`: we would apply rolling friction to gear play.
    The roller env tightened the regex to `^passive_.*wheel`; this variant must
    inherit that.
    """
    bl = _load("Mjlab-RollerStandUp-Flat-Backlash-MicroDuck")
    names = bl.events["randomize_wheel_friction"].params["asset_cfg"].joint_names
    assert all("wheel" in n for n in names), names
    # And the starting value stays the inverted curriculum's stage 0.
    stage0 = bl.curriculum["wheel_friction"].params["ranges_stages"][0]["ranges"]
    assert bl.events["randomize_wheel_friction"].params["ranges"] == stage0


# ── Support gate: "standing" means standing ON THE WHEELS ────────────────────
# Fix for run fmt83tri, where the policy converged to a head tripod (head on the
# floor, trunk levered to the right height at 55° of tilt) that collected 48 % of
# the maximum task stack without ever standing up.


def test_support_gate_sensors_are_declared():
    cfg = make_microduck_roller_standup_env_cfg()
    names = {s.name for s in cfg.scene.sensors}
    # Inherited from the roller env.
    assert "feet_ground_contact" in names
    assert "self_collision" in names
    # Added for the support gate. ALL THREE are required: without
    # limbs_ground_contact, a robot sprawled on hips + shins opens the gate.
    assert "head_ground_contact" in names
    assert "trunk_ground_contact" in names
    assert "limbs_ground_contact" in names


def test_trunk_ground_sensor_is_body_not_subtree():
    """trunk_base's subtree contains the TIRES.

    With mode="subtree", trunk_ground_contact would be permanently true — including
    while standing on the wheels — so the gate would stay closed and ALL gated
    rewards would be zero forever. That is a silent failure mode: nothing would
    crash, the policy would simply never learn.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    trunk = next(s for s in cfg.scene.sensors if s.name == "trunk_ground_contact")
    assert trunk.primary.mode == "body"
    assert trunk.primary.pattern == "trunk_base"


def test_gated_sensor_bodies_exist_on_both_roller_models():
    """The targeted bodies must exist on the rollers AND rollers+backlash models.

    Same reason as the joint-index test: a sensor resolving no body does not
    announce itself in a cfg, only at runtime.
    """
    import mujoco

    from mjlab_microduck.robot.microduck_constants import (
        get_rollers_backlash_spec,
        get_walk_rollers_spec,
    )

    for spec_fn in (get_walk_rollers_spec, get_rollers_backlash_spec):
        model = spec_fn().compile()
        bodies = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)
        }
        for name in ("jaw_soft", "trunk_base", "hip_l", "hip_l_2", "leg", "leg_2"):
            assert name in bodies, f"{spec_fn.__name__}: {name} missing"


def test_goal_state_rewards_are_gated_on_wheel_support():
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_microduck_roller_standup_env_cfg()
    assert (
        cfg.rewards["pose_stand_legs"].func
        is microduck_mdp.pose_target_match_on_wheels
    )
    assert (
        cfg.rewards["standing_composite"].func
        is microduck_mdp.standing_composite_score_on_wheels
    )


def test_climb_shaping_stays_ungated():
    """Only the GOAL-STATE payout is gated, not the climb shaping.

    If height_stand / height_stand_l1 / upright_linear were gated too, nothing
    would pull the robot off the floor any more: from a ground pose the gate is
    closed, so the entire gradient would vanish. That is the "motion-blocker"
    failure mode under another name.
    """
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.rewards["height_stand"].func is microduck_mdp.height_target_gaussian
    assert cfg.rewards["height_stand_sharp"].func is microduck_mdp.height_target_gaussian
    assert cfg.rewards["height_stand_l1"].func is microduck_mdp.height_l1_penalty
    assert cfg.rewards["upright_linear"].func is microduck_mdp.body_upright_linear
    # height_progress is the term that carries the gradient off the floor: gating
    # it would defeat its entire purpose.
    assert cfg.rewards["height_progress"].func is microduck_mdp.height_progress


def test_height_progress_is_present_and_capped_at_the_roller_stand():
    """The gradient the back start lacks, and the reason it is a Δ and not a level.

    Measured on the supine→prone roll sweep: the whole rollover is worth
    +0.156/step and its first half is downhill (on the side the trunk sits 7 mm
    lower than on the back, so height_stand_l1 penalises starting the move).
    Every gated term reads 0.0000 across the full 180°.

    height_progress pays Δ min(z, ceiling): rising pays, holding pays exactly
    zero, falling refunds. A level-based reward (a wider height_stand Gaussian)
    would buy the same floor gradient at the price of paying 0.267/step for
    merely lying on the back — a free floor, the mechanism behind the tripod.

    It measures HEIGHT, not posture, so it prescribes no technique.
    """
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import ROLLER_STAND_Z

    cfg = make_microduck_roller_standup_env_cfg()
    term = cfg.rewards["height_progress"]
    # Positive weight: the function returns a signed Δ, so rising must pay.
    assert term.weight > 0
    # Capped at the standing height: hopping higher must pay nothing extra.
    assert term.params["ceiling"] == ROLLER_STAND_Z


def test_height_progress_weight_cancels_the_action_tax():
    """The weight is derived, not guessed — keep the derivation checkable.

    Full rise 0.046 → 0.138 = 0.092 m. At weight w the climb collects w·0.092,
    and a 2 s climb (100 steps, 0.92 mm/step) pays w·0.00092 per step. The target
    is to cancel the ~-0.2/step that action_rate_l2 charges a smooth policy at
    its full -1.0 weight, i.e. w ≈ 217.

    velstand runs the same function at weight 30, but there it is a last-mile
    helper on top of a working recovery; here it must be the primary gradient
    across a 9 cm unpaid region. Being potential-based (Ng et al.), it is
    policy-invariant — a large weight cannot create a new optimum.
    """
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import ROLLER_STAND_Z

    cfg = make_microduck_roller_standup_env_cfg()
    w = cfg.rewards["height_progress"].weight
    per_step = w * 0.00092
    assert 0.1 <= per_step <= 0.5, (
        f"weight {w} pays {per_step:.3f}/step on a 2 s climb; the point is to be "
        f"the same order as action_rate_l2's ~-0.2/step"
    )
    # And the full climb must stay small against the standing payout (~10/step),
    # so the shaping stays a nudge rather than a destination of its own.
    full_climb = w * (ROLLER_STAND_Z - 0.046)
    assert full_climb < 40.0, f"full climb pays {full_climb:.1f}, too close to a jackpot"


def test_composite_weight_unchanged_by_the_gate():
    """One fix at a time: the gate, not the gate AND the weight."""
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.rewards["standing_composite"].weight == 3.75
    assert cfg.rewards["pose_stand_legs"].weight == 2.0


def test_backlash_variant_keeps_the_support_gate():
    from mjlab_microduck.tasks import mdp as microduck_mdp

    bl = _load("Mjlab-RollerStandUp-Flat-Backlash-MicroDuck")
    names = {s.name for s in bl.scene.sensors}
    assert "head_ground_contact" in names
    assert "trunk_ground_contact" in names
    assert "limbs_ground_contact" in names
    assert (
        bl.rewards["standing_composite"].func
        is microduck_mdp.standing_composite_score_on_wheels
    )
