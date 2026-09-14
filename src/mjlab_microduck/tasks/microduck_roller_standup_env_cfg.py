"""Microduck roller standup — getting back up on roller skates.

DEDICATED episodic policy: the robot starts on the ground (face-down, face-up) or
already standing, and must get back up onto its wheels and then HOLD the stand.
Port of the `standup` recipe (walking duck) onto the rollers model.

Derives from the roller env (`make_microduck_velocity_rollers_env_cfg`) → inherits
the rollers robot, the sensors, the whole DR stack and the 61D observation as-is,
so it stays runtime-swappable (--new-cmd-obs). Same pattern as roller_slope.

One structural difference from `standup`: no head_pose command — the head/body
slots stay zero-padded (roller-family convention) and the head is held upright by
neck_joint_pos_l2, which resolves by NAME. Joint indices follow the canonical
14-servo layout, same as the walker (see _LEG_JOINTS): mdp indexes through
_servo_joint_pos, which excludes wheels and backlash hinges.

Reward weights are aligned on the evolved standup recipe (task block at 1/4,
anti-violence penalties introduced only at iteration 3000). See
docs/roller_standup_policy_summary.md for the failure history behind them.

The genuinely new piece is the rolling-friction curriculum, INVERTED (braked
wheels → free wheels): the wheels roll, so there is no longitudinal grip to push
against the floor. We bootstrap with nearly-locked wheels then ramp down to the
real value. If `standing_composite` collapses at a stage, the "grippy feet"
gesture does not transfer and a skater technique would have to be guided (knee
support, one skate at a time).

Target deployment: `--standing` alongside the roller policy in `--walking`, with
the automatic switch on velocity-command magnitude (infer_policy.py:262,
threshold 0.05); the twist slot is left at zero there (infer_policy.py:239).
"""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# ── Trunk heights (m) ─────────────────────────────────────────────────────────
# Measured by exact kinematics (lowest mesh vertex of the colliding geoms, STAND
# pose, trunk lowered to contact) on scene_rollers.xml: standing 0.1407,
# face-down rest 0.0752, face-up rest 0.0475.
# The 0.1407 → 0.138 step borrowed the ~2 mm of load sag measured on the model
# WITHOUT wheels (kinematic 0.1172 vs STAND_Z=0.115 measured under load by
# standup). That borrow has since been VERIFIED on the rollers model itself:
# 512 envs spawned standing, HOME ctrl, DR off, envs at tilt < 5° only settle to
# z = 0.1386 (+0.6 mm from target), i.e. 2.1 mm of real sag against 2.2 mm
# borrowed. height_stand_sharp has a 15 mm std, so 1 mm of target error costs
# 0.4 % of the reward — no effect.
# 0.138 also falls inside the reset_base z band (0.1335–0.1435) the roller env
# already uses.
ROLLER_STAND_Z = 0.138
ROLLER_PRONE_Z = 0.075

EPISODE_LENGTH_S  = 6.0   # rise + stabilise, same as standup
NUM_STEPS_PER_ENV = 24

# ── Play override: force the share of FACE-UP (on the back) spawns ────────────
# At play time the env is rebuilt from scratch: common_step_counter restarts at
# 0, so the ground_state_mix curriculum applies its stage 0, where
# face_up_prob = 0. A play therefore NEVER shows a back start — yet that is the
# hardest case, the one worth inspecting by eye. This variable forces it.
#   STANDUP_PLAY_FACE_UP=1.0  -> 100 % back starts
#   STANDUP_PLAY_FACE_UP=0.4  -> the final curriculum stage's mix
#   unset / "none" / "random" -> default behaviour (stage 0)
# Affects play=True ONLY. Same pattern as SLOPE_PLAY_DIFFICULTY in roller_slope.
PLAY_FACE_UP = None
# face-down:standing ratio of the LAST curriculum stage (0.40 / 0.20 = 2:1). The
# remainder (1 - face_up) is split in that ratio, so 0.4 reproduces the
# end-of-training mix exactly.
_PLAY_FACE_DOWN_SHARE = 2.0 / 3.0


# ── Play override: force the wheel ROLLING FRICTION ───────────────────────────
# Same mechanism as PLAY_FACE_UP, and just as load-bearing. A play env is rebuilt
# from scratch, so common_step_counter restarts at 0 and wheel_friction_curriculum
# applies its stage 0 — 0.05, i.e. nearly locked wheels. Every play session
# therefore shows a robot whose wheels do not turn, whatever the checkpoint.
#
# Measured at 32 envs, standing spawns, HOME ctrl:
#
#   frictionloss   |ω wheel| max, free settle   under a 0.5 m/s push
#   0.0500 (stage 0)          0.29 rad/s              0.40 rad/s
#   0.0015 (stage 4)         23.18 rad/s             24.83 rad/s
#
# A factor of 85. A post-4000 checkpoint trains at 0.0015 and can only be judged
# by eye at 0.0015; watching it at stage 0 shows a behaviour it was never trained
# for, and hides every sliding/rolling artefact that shows up on the real robot.
#
#   STANDUP_PLAY_WHEEL_FRICTION=0.0015  -> the real rolling value (stage 4)
#   STANDUP_PLAY_WHEEL_FRICTION=0.05    -> stage 0, the bootstrap value
#   unset / "none" / "curriculum"        -> default (stage 0, as before)
#
# play=True ONLY — training and its curriculum are untouched.
PLAY_WHEEL_FRICTION = None


def _resolve_play_wheel_friction():
    """Wheel frictionloss forced at play time, or None to keep the curriculum."""
    raw = os.environ.get("STANDUP_PLAY_WHEEL_FRICTION")
    if raw is None:
        return PLAY_WHEEL_FRICTION
    raw = raw.strip().lower()
    if raw in ("", "none", "curriculum"):
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        print(
            f"[roller_standup] STANDUP_PLAY_WHEEL_FRICTION='{raw}' invalid "
            f"-> default {PLAY_WHEEL_FRICTION}"
        )
        return PLAY_WHEEL_FRICTION


def _resolve_play_face_up():
    """Share of back starts at play: STANDUP_PLAY_FACE_UP env var, else the constant."""
    raw = os.environ.get("STANDUP_PLAY_FACE_UP")
    if raw is None:
        return PLAY_FACE_UP
    raw = raw.strip().lower()
    if raw in ("", "none", "random"):
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        print(f"[roller_standup] STANDUP_PLAY_FACE_UP='{raw}' invalid -> default {PLAY_FACE_UP}")
        return PLAY_FACE_UP

# ── Joint indices — canonical SERVO-ONLY layout (14 joints) ───────────────────
# pose_target_match / pose_l1_penalty / standing_composite_score index through
# mdp._servo_joint_pos, which selects `^(?!passive_).*`: the 14 servos, excluding
# ALL passive joints — wheels AND backlash hinges. Indices are therefore written
# in that 14-joint view, identical for the walker and for the rollers:
#   0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
#   5-8   neck_pitch, head_pitch, head_yaw, head_roll
#   9-13  right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
#
# ⚠️ HISTORY — do NOT "fix" these back to full-array positions.
# This file used [0-4, 11-15] / [7-10], the real positions in the rollers
# entity's 18-joint array (wheels interleaved at 5,6 and 16,17). That was correct
# back when rewards indexed asset.data.joint_pos directly. Since mdp.py migrated
# to _servo_joint_pos, indices 14 and 15 fall outside a 14-column tensor →
# "index out of bounds" on GPU, env untrainable. Config tests do not catch it:
# they only build cfg, never call the rewards. Only a real run reveals it.
#
# Benefit of the servo-only view: it is IDENTICAL on the rollers model and on the
# rollers+backlash model (32 joints, 18 of them passive), so these indices are
# now backlash-proof. Checked on both models by
# tests/test_roller_standup_cfg.py::test_joint_indices_are_in_the_canonical_servo_space.
#
# Only _LEG_JOINTS is consumed (pose rewards). _NECK_JOINTS serves documentation
# and the test: the neck resolves by NAME (neck_joint_pos_l2 calls
# find_joints(r".*(neck|head).*") every step). No more _WHEEL_JOINTS: the wheels
# do not exist in the servo-only view, and their DR targets them through the
# `^passive_.*wheel` regex.
_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# SKATING rewards from the roller env: meaningless while lying on the floor.
# feet_flat: the blades are NOT flat during the rise → would fight the gesture.
# hip_roll_neutral: getting up requires spreading the legs.
# pose / com_height_target: replaced by the standup pose/height targets.
# upright (base Gaussian): replaced by upright_linear + upright_sharp.
_SKATING_REWARDS = (
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


def make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """"Get up on roller skates" env: start on the ground, target = standing on wheels."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Contact sensors for the SUPPORT GATE ─────────────────────────────────
    # The roller env only declares feet_ground_contact (tires/terrain) and
    # self_collision. We additionally need to know whether the HEAD, the TRUNK or
    # the LIMBS touch the floor: without that, "upright at the right height"
    # cannot tell standing on wheels apart from the head tripod measured on run
    # fmt83tri (see mdp.wheel_support_gate for the numbers).
    #
    # The first two definitions are ported as-is from
    # microduck_roulade_env_cfg.py and are valid on the rollers model: `jaw_soft`
    # carries the head collision geoms (top_head_shell, bottom_head_shell, jaw)
    # and `trunk_base` carries np_f970 (the battery), the trunk part that touches
    # the floor when lying flat.
    # mode="body" and NOT "subtree" for trunk_base: the subtree would contain the
    # tires, so the gate would be permanently closed, including while standing.
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_contact",
        primary=ContactMatch(mode="body", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # Hips and shins — the measured hole in the v1 gate. The model carries only
    # 12 COLLISION geoms: the battery (trunk_base), 3 head geoms (jaw_soft),
    # hip_l/hip_l_2, leg/leg_2, and the 4 tires. The trunk shells are
    # VISUAL-only, so a robot sprawled on its hips and shins, one tire grazing
    # the floor and the head held up, opened neither of the two sensors above:
    # the gate opened while lying flat on the ground, and `pose_stand_legs` paid.
    # With this third sensor, "gate open" means exactly "only the tires touch the
    # floor".
    limbs_ground_cfg = ContactSensorCfg(
        name="limbs_ground_contact",
        primary=ContactMatch(
            mode="body", pattern=r"^(hip_l|hip_l_2|leg|leg_2)$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        head_ground_cfg,
        trunk_ground_cfg,
        limbs_ground_cfg,
    )

    # ── Skating rewards removed ──────────────────────────────────────────────
    for name in _SKATING_REWARDS:
        cfg.rewards.pop(name, None)

    # ── Attempt tax: dropped, matching the reference recipe ──────────────────
    # neck_action_rate_l2 (-0.5) taxes head MOTION. The 4 head joints are already
    # covered by action_rate_l2, so this term DOUBLE-taxes them: effective weight
    # per head joint 0.6 against 0.1 per leg joint at stage 0, i.e. 6x at the
    # exact moment of discovery.
    #
    # Measured on the smoke (a moving policy): -1.359/step, the largest term in
    # the whole reward, ~1.9x the entire positive task block. The arithmetic the
    # policy then faces from face-down: stay still ≈ -0.39/step, move ≈
    # -4.2/step. Doing nothing wins by 10x — AGENTS.md's law on attempt taxes
    # during discovery, in numbers.
    #
    # The walker standup DROPS it explicitly (microduck_standup_env_cfg.py:485,
    # "microduck-only extras DROPPED, like velocity drops them"). It arrived here
    # by inheritance from the SKATING recipe, where a calm head serves a gait; it
    # was never audited for a standup.
    #
    # ⚠️ Do NOT confuse it with neck_joint_pos_l2 (-0.5), which is KEPT: that one
    # taxes the head being FAR FROM NEUTRAL, so it fights the tripod. The two
    # terms pull in opposite directions and the RATE one is the tax.
    #
    # action_over_limit (-0.5) is kept as well: it penalises commands outside
    # ctrlrange, a specific pathology and the roller family's policy-side sim2real
    # guard — not a tax on motion.
    cfg.rewards.pop("neck_action_rate_l2", None)

    # ── Command: twist slot neutralised (≈ 0) ────────────────────────────────
    # The roller env installs a RelativeHeadingVelocityCommandCfg (cmd[2] =
    # heading error computed internally). Here nothing is steered: we go back to
    # the neutralised command-only variant, like standup. The head_pose (4) and
    # body_pose (6) slots stay zero-padded → 61D obs parity preserved.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Numerical robustness (same choice as roller_slope) ───────────────────
    # A rare contact (~1/25M steps) diverges the free joint into NaN: we sanitise
    # the obs (→ 0) rather than kill training, and the offending env resets on the
    # next step.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    # ── Standup rewards — transplanted from standup, remapped ────────────────
    # The weights come from the iterations documented in
    # microduck_standup_env_cfg.py: only touch them with a reason. Only the joint
    # indices and the two heights change here.
    # NB: a FRESH SceneEntityCfg per term — mjlab resolves and mutates them in
    # place, so a shared object yields stale indices.

    # Target pose = HOME (target_overrides=None), LEGS only: the neck and head
    # are held by neck_joint_pos_l2 (inherited), which resolves by NAME.
    # ⚠️ The whole task block below sits at 1/4 of the original weights, aligned
    # on the evolved standup recipe. The dampers keep their values, so dividing
    # the task fixes the task/damper ratio — measured at ~35:1 (task ≈ +41.6
    # against ≈ -1.2 of dampers, run vweolw91), which left no reason to be gentle.
    # Dividing the task rather than raising the dampers avoids turning one of them
    # into a motion-blocker. The ratios INTERNAL to the block are unchanged, as are
    # all the stds.
    # ⚠️ SUPPORT-GATED (fix for run fmt83tri). The ungated version read
    # 1.991/2.000 from iteration 250 to 3625 without ever moving — the legs sit
    # near HOME in EVERY posture the policy visits, so it was an unconditional
    # +2.0/step with no gradient, and 40 % of what the tripod kept. The gate
    # preserves the term's purpose (hold the legs near HOME once up) and removes
    # the free floor.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match_on_wheels,
        weight=2.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )
    # L1 bootstrap: constant gradient even far from HOME (the Gaussian saturates).
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=1.25,
        params={
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # Height in three layers: wide Gaussian (pulls up from the floor), narrow
    # Gaussian (forces the last centimetres, where the wide one is saturated),
    # and a strong L1 that makes "stay on the ground" net NEGATIVE — without it
    # the policy settles for the lazy optimum "motionless on the floor".
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std": 0.04,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std": 0.015,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=7.5,
        params={
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # ── Potential-based height progress — the gradient the back start lacks ──
    # MEASURED PROBLEM (run at iter 2500: face-down standup works, the back never
    # moves). Sweeping the supine→prone roll by 10° steps, 8 envs per angle, the
    # total reward reads:
    #
    #     roll   0° (flat on back)   -0.719     <- every gated term is 0.0000
    #     roll  90° (on the side)    -0.757     <- WORSE than staying on the back
    #     roll 180° (face down)      -0.563
    #
    # So the entire rollover is worth +0.156/step, and its first half is
    # DOWNHILL: on the side the trunk sits 7 mm lower than on the back, so
    # height_stand_l1 actively penalises starting the move. Meanwhile moving
    # costs action_rate_l2 at its full -1.0 (~-0.2/step for a smooth policy).
    # Doing nothing is simply the better deal, whatever technique the policy
    # might otherwise invent.
    #
    # This also explains why face_up_roll_max alone did not unlock the back: a
    # reverse curriculum supplies on-policy DATA, it does not create a gradient,
    # and the data said rolling does not pay.
    #
    # WHY THIS TERM AND NOT A WIDER height_stand GAUSSIAN: widening the std to
    # 0.08 would give the same gradient at floor level, but it pays 0.267/step
    # for merely LYING on the back (against 0.005 today) while the standing robot
    # gains nothing — the standing/lying gap shrinks and a free floor reappears.
    # Any reward that pays for BEING at a height necessarily pays for being there
    # effortlessly. This one pays for GAINING height, so holding any pose is
    # worth exactly zero.
    #
    # TECHNIQUE-AGNOSTIC ON PURPOSE: it measures trunk height, not posture. Roll,
    # pike, pivot on a shoulder, or anything nobody has thought of — all paid the
    # same, per centimetre gained. The path stays what RL is supposed to discover.
    #
    # Potential-based (Ng et al.), so it is policy-invariant: it cannot create a
    # new optimum, which is what makes a large weight safe here.
    #
    # WEIGHT 200, derived rather than guessed:
    #   full rise 0.046 → 0.138 = 0.092 m  →  total +18.4 over the climb
    #   a 2 s climb (100 steps) ≈ 0.92 mm/step  →  +0.18/step
    # which roughly cancels the -0.2/step that action_rate_l2 charges a smooth
    # policy. That is the target: make progress at a plausible climb rate pay for
    # its own action cost. velstand runs the same function at weight 30, but
    # there it is a last-mile helper on top of a working recovery; here it has to
    # be the primary gradient across a 9 cm unpaid region.
    #
    # Ceiling at ROLLER_STAND_Z so hopping above standing height pays nothing
    # extra. NOT support-gated, deliberately: the floor is exactly where the
    # signal is needed.
    cfg.rewards["height_progress"] = RewardTermCfg(
        func=microduck_mdp.height_progress,
        weight=200.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "ceiling": ROLLER_STAND_Z,
        },
    )

    # Pays for the MOTION of rising, not only the destination: without it,
    # "stay seated collecting the partial pose" dominates. The cutoff sits 10 mm
    # ABOVE the target, otherwise the policy parks at the cutoff altitude and
    # never finishes the climb.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.75,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": ROLLER_STAND_Z + 0.010,
        },
    )
    # Gentle rise: penalises |a_z|. Compatible with com_upward_velocity — a
    # constant vertical velocity collects the latter AND has a_z = 0, so the two
    # pressures jointly select a smooth constant-speed rise.
    #
    # ⚠️ POSITIVE WEIGHT, and it is not a typo. mdp.py mixes two sign
    # conventions: trunk_vertical_accel_penalty already returns -|a_z|
    # (mdp.py:2171), like height_l1_penalty and pose_l1_penalty — which are used
    # here with positive weights too. The -0.02 inherited from standup was
    # therefore a double negative and REWARDED vertical acceleration: measured at
    # Episode_Reward/gentle_rise = +0.0118 (the only penalty term logging
    # positive) on run vweolw91. That is the cause of the "very violent"
    # behaviour, and it also explains the fruitless damping attempts documented
    # in standup, which were fighting a term actively pushing the other way.
    #
    # Magnitude 0.005, the CEILING measured by standup: "the 2026-07-24 attempt to
    # double it to -0.01 contributed to the face-up freeze; -0.005 is the ceiling
    # unless it gets a height/tilt gate like arrival_damping". This term is GLOBAL
    # (ungated), so ground rollovers pay it in full — hence the ceiling.
    # Deliberately small: |a_z| is necessarily high during a rollover from the
    # back, so a large weight here would be a motion-blocker. Real damping is
    # carried by joint_torque_rate_l2, which penalises torque VARIATION rather
    # than motion.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # ARRIVAL damper — trunk ω_xy², gated on height AND tilt.
    # Targets exactly the failure loop described on the robot: rise → overshoot
    # vertical → tip → retry. That is what "violence" means here.
    #
    # Height gates transposed onto ROLLER_STAND_Z: standup uses 0.09/0.11 for
    # STAND_Z=0.115, i.e. -25 mm and -5 mm below the stand. Copied verbatim onto
    # the roller (0.138) they would open the gate while the robot is still ~3 cm
    # below its standing height, i.e. MID-RISE — precisely what the gate must
    # spare.
    #
    # The TILT gate is essential, not a refinement: without it, the final
    # straightening of a folded rise (tilt 60°→0 inside the height gate) is
    # itself a large trunk rotation; taxing it raises a reward wall just before
    # arrival and the policy parks folded under the gate.
    #
    # STARTS AT 0 — introduced at iteration 3000 by arrival_damping_weight. See
    # the curriculum block for the timing lesson, which is the real cause of this
    # env's freeze.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low":    ROLLER_STAND_Z - 0.025,
            "height_high":   ROLLER_STAND_Z - 0.005,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Upright trunk in two layers: cos(tilt) has a strong gradient while lying
    # down but runs out of steam near vertical; the tight height-gated Gaussian
    # takes over and kills the backward lean (standup's failure mode: tipping
    # backwards by extending the legs).
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=1.5,
        params={
            "std": 0.3,
            "height_low": ROLLER_PRONE_Z,
            "height_high": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # MULTIPLICATIVE height × uprightness × pose score: because the factors
    # multiply, being good on 2 criteria out of 3 pays nothing → it breaks the
    # "leaning at the right height" compromises that additive rewards let
    # through. Stds deliberately WIDE so the score stays visible during the rise
    # (tight stds scored ~5e-5, i.e. zero gradient).
    # ⚠️ SUPPORT-GATED as well. The multiplicative composite was doing its job
    # (0.915/3.75, i.e. 24 % — it did collapse on the uprightness factor), but it
    # only weighs 3.75 of a 10.75 positive mass: a compromise that keeps 48 % of
    # the stack still wins. A multiplicative score cannot break a compromise that
    # the REST of the stack finances.
    #
    # The WEIGHT IS UNCHANGED, deliberately: the gate alone takes the tripod from
    # 5.16 to 2.245 out of 10.75 (48 % -> 21 %), so the standing/sprawling
    # differential goes from 2.1x to 4.8x. Raising the weight at the same time
    # would make the result unattributable — the method lesson this env has
    # already paid for once (three fixes at once, unexplainable freeze).
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score_on_wheels,
        weight=3.75,
        params={
            "target_height": ROLLER_STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Anti-jitter: penalises torque VARIATION, not its magnitude nor trunk
    # rotation → damps the shakes without blocking the rollover. standup
    # identified it as the only damper that does not kill back recovery, so it is
    # THE safe lever to raise.
    #
    # -2e-3 (the value inherited from standup) contributed only -0.0002/step
    # against ~+41.6 of task reward saturated at 95-99 % — i.e. nothing at all.
    # Across all dampers the ratio was ~35:1 in favour of the task, so there was
    # no reason to be gentle. Measured on run vweolw91 at iteration 7500.
    #
    # Recalibration: the raw |Δτ|² value is ~0.1 at convergence, so the
    # contribution ≈ 0.1 × |weight|. Measured at -0.255/step with weight -2.0 (run
    # d8rnko6p) — so NOT the cause of the freeze, but we come back down to -0.2 to
    # free up the damping budget while the sign bug's effect is isolated.
    # If it is still violent, raise THIS term (formula above) rather than
    # body_ang_vel or action_rate, which are motion-blockers and were freezing
    # back recovery.
    # STARTS AT 0 — introduced at iteration 3000 by torque_rate_weight. At -0.2
    # from step 0 (run d8rnko6p) it was part of the attempt tax that froze the
    # policy. standup ended up setting it to -1e-3 only, and only after 3000:
    # arrival damping is carried by arrival_damping, which is gated and therefore
    # surgical.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=0.0,
    )

    # NO head-impact penalty. Tried with velstand's values (body_impact_cost,
    # `neck` subtree, weight -1.0, threshold 2.0): the policy converged to lying
    # down, INERT. Measured (run d8rnko6p): head_impact_penalty -1.01/step, the
    # largest negative term in the table, while standing_composite collapsed from
    # +14.3 to +3.3.
    #
    # The reasoning error was believing that a "targeted" penalty does not
    # restrict motion. False here: to get up from its back, this robot PIVOTS on
    # its head and shoulders. The head is the rollover's support point, not
    # collateral damage — penalising it blocks the only available mechanism, and
    # the back was already the failing case.
    #
    # Hypothesis under test: banging the head was a SYMPTOM of the violence (the
    # gentle_rise sign bug paid for brutality, and a brutal rise ends on the
    # head), not a separate defect. If the slam comes back now that the sign is
    # fixed, the reintroduction must be a HEIGHT-GATED penalty — the way
    # upright_sharp is — so the ground rollover phase is spared.
    #
    # ⚠️ Watch the lazy optimum that makes such a freeze possible:
    # pose_stand_legs stayed at +7.72 out of 8 while the robot was lying flat
    # (legs at HOME in a lying pose → reward collected almost for free). That is
    # what height_stand_l1 (weight +30) had to make net negative — and what the
    # support gate now handles directly.

    # ── GROUND start: face-down / face-up / already standing ─────────────────
    # Added LAST in cfg.events: execution order follows insertion order, and this
    # term must overwrite the pose set by reset_base / reset_robot_joints.
    # The "already standing" bucket is not decorative: without it the policy
    # learns to rise but not to HOLD, and falls right after getting up.
    # No "sitting" bucket → no sitting_joint_overrides to remap (standup's are
    # indices of the model WITHOUT wheels).
    # The probabilities below = stage 0 of the ground_state_mix curriculum.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,   # face-down (+90° pitch)
            "face_up_prob":   0.00,   # on the back — hardest, introduced late
            "sitting_prob":   0.00,
            "standing_prob":  0.50,
            "sitting_joint_overrides": None,
            # Both ground poses (face-down/face-up) share a SINGLE z range, yet
            # their contacts have nothing in common: the belly only clears the
            # floor from 0.0752, the back rests at 0.0475. A single floor value
            # therefore cannot be ideal for both. We pick 0.076 to eliminate any
            # interpenetration on the face-down side (measured: at 0.05, +25 mm
            # into the ground), at the cost of a back start 28–42 mm above its
            # rest height — a far milder artefact than a contact pushout.
            "prone_z_min":    0.076,
            "prone_z_max":    0.09,
            # Standing on wheels: ROLLER_STAND_Z = 0.138 (vs 0.11–0.12 without).
            "standing_z_min": 0.134,
            "standing_z_max": 0.144,
            # Pitch/roll noise at spawn. Careful: inside set_random_ground_state
            # the "standing" bucket reuses the "sitting" bucket's quaternion, so
            # this noise ALSO applies to standing starts — which is intended (no
            # overfitting to perfectly-upright starts).
            "sitting_tilt_max": math.radians(10),
            # ── Built-in reverse curriculum on back starts ────────────────────
            # Roll noise (±90° about the body's LONG axis) applied to the face_up
            # bucket only. This parameter was missing — so it defaulted to 0, and
            # ALL back starts were perfectly flat.
            #
            # That is the case the walker documents as hopeless: between flat
            # supine and prone the reward landscape is FLAT — upright_linear
            # (cos tilt) stays ≈ 0 through the whole roll and the height does not
            # change — so rolling only pays via the frontal rise that follows, a
            # long-horizon dependency that noisy exploration almost never finds
            # from a perfectly flat supine start. Its verdict: back recovery was
            # "seed-lucky", 1 success for 3 failures at equivalent rewards.
            #
            # Measured here on 256 spawns: tilt reads 90.0° for all of them
            # whatever the roll — rolling about the long axis does not change how
            # far the trunk is from vertical. Direct confirmation that no
            # gradient guides the roll.
            #
            # With the noise, a fraction of starts begins nearly on the side,
            # i.e. part-way through the roll: the policy learns the END of the
            # gesture from easy starts then generalises back to flat supine.
            # Built-in reverse curriculum, with no stage to tune — uniform
            # sampling keeps every difficulty represented at all times (nearly
            # flat back, |roll| < 15°, ≈ 17 % of draws at ±90°).
            #
            # Interaction with prone_z_min=0.076: a side start rests lower than a
            # flat back, so it drops a few mm at spawn (measured: 42 mm median
            # drop, 0/256 envs interpenetrating). That is the safe direction of
            # error — a fall, not an interpenetration, which is exactly what
            # prone_z_min=0.076 exists to avoid.
            "face_up_roll_max": math.radians(90),
        },
    )

    # The robot STARTS fallen → a tilt termination makes no sense here (it would
    # kill the episode on the first step). nan_state, inherited, stays.
    cfg.terminations.pop("fell_over", None)

    # Start-pose curriculum, easy → hard. With a flat mix from the start, the
    # policy optimises the easy majority and leaves the back under-trained
    # (standup's lesson: it froze into "do nothing" on that pose). So standing +
    # face-down come first, the back comes late, and the mix is biased toward the
    # hard poses at the end so they get the most practice.
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                {"step": 0, "params": {
                    "standing_prob": 0.50, "sitting_prob": 0.00,
                    "face_down_prob": 0.50, "face_up_prob": 0.00}},
                {"step": 600 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.35, "sitting_prob": 0.00,
                    "face_down_prob": 0.45, "face_up_prob": 0.20}},
                {"step": 1500 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.25, "sitting_prob": 0.00,
                    "face_down_prob": 0.40, "face_up_prob": 0.35}},
                {"step": 2500 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.20, "sitting_prob": 0.00,
                    "face_down_prob": 0.40, "face_up_prob": 0.40}},
            ],
        },
    )

    # Play override: force back starts so they can be inspected. We write the
    # probabilities into the event AND remove the curriculum: without that,
    # event_param_curriculum (which runs BEFORE the reset events) would rewrite
    # them with its stage 0 on the very first reset. play only, so training and
    # its easy → hard curriculum are untouched.
    if play:
        play_face_up = _resolve_play_face_up()
        if play_face_up is not None:
            remainder = 1.0 - play_face_up
            cfg.events["set_ground_state"].params.update({
                "face_up_prob":    play_face_up,
                "face_down_prob":  remainder * _PLAY_FACE_DOWN_SHARE,
                "standing_prob":   remainder * (1.0 - _PLAY_FACE_DOWN_SHARE),
                "sitting_prob":    0.00,
            })
            del cfg.curriculum["ground_state_mix"]

    # ── INVERTED rolling friction: braked → free ─────────────────────────────
    # This was meant to be the genuinely new piece of this env, and its supposed
    # core difficulty: the wheels roll, so there is NO longitudinal grip to push
    # against the floor. The roller env RAISES this friction (0 → 0.0015); here
    # we LOWER it, to bootstrap the gesture on an easy problem (nearly-locked
    # wheels ≈ feet) before imposing the real rolling physics.
    #
    # ⚠️ PARTIAL MEASUREMENT, DO NOT ACT ON IT YET (run fmt83tri): at all three
    # stages (1000/2000/3000, i.e. 0.05 → 0.003, a factor 17) standing_composite
    # did not drop — at 2000 and 3000 it even rose just after.
    #
    # But that run NEVER STOOD UP: its composite tracked standing_prob, and its
    # behaviour was a head tripod, which does not carry the body on the wheels at
    # all. A friction that only matters to a wheel-supported rise could not have
    # shown up there. The earlier note in this file calling the curriculum a
    # "non-event" and the wheel hypothesis "refuted" was therefore premature.
    #
    # The first real test is the run where the face-down standup works: stages
    # 3000 (0.003) and 4000 (0.0015) are then the ones to watch. Do not compress
    # this curriculum before that evidence exists.
    #
    # sim2real note: only checkpoints AFTER the last stage (iter 4000+) are
    # deployment candidates. Before that the policy leans on a rolling friction
    # that does not exist on the real robot.
    _WHEEL_FRICTION_STAGE0 = (0.0500, 0.0500)
    cfg.curriculum["wheel_friction"] = CurriculumTermCfg(
        func=microduck_mdp.wheel_friction_curriculum,
        params={
            "event_name": "randomize_wheel_friction",
            "ranges_stages": [
                {"step": 0,                        "ranges": _WHEEL_FRICTION_STAGE0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "ranges": (0.0200, 0.0200)},
                {"step": 2000 * NUM_STEPS_PER_ENV, "ranges": (0.0080, 0.0080)},
                {"step": 3000 * NUM_STEPS_PER_ENV, "ranges": (0.0030, 0.0030)},
                {"step": 4000 * NUM_STEPS_PER_ENV, "ranges": (0.0015, 0.0015)},
            ],
        },
    )
    # Defensive redundancy: the curriculum manager runs BEFORE the reset events on
    # every reset (including the very first), and wheel_friction_curriculum itself
    # defaults to stage 0 — so this line is never needed in practice. It merely
    # keeps the event's DEFAULT value consistent with the curriculum's stage 0, in
    # case someone later removes the curriculum and leaves the event in place.
    cfg.events["randomize_wheel_friction"].params["ranges"] = _WHEEL_FRICTION_STAGE0

    # Play override — MUST come after the curriculum is defined above, otherwise
    # the `del` hits a missing key and the curriculum is recreated right after.
    # Both halves are required for the same reason the face-up override needs
    # both: the curriculum runs BEFORE the reset events, so writing the event
    # alone is silently overwritten with stage 0 on the very first reset. That is
    # not a theory — it defeated the first attempt at measuring the wheel spin,
    # which reported 0.29 rad/s instead of 23 rad/s.
    if play:
        play_wheel_friction = _resolve_play_wheel_friction()
        if play_wheel_friction is not None:
            cfg.events["randomize_wheel_friction"].params["ranges"] = (
                play_wheel_friction,
                play_wheel_friction,
            )
            del cfg.curriculum["wheel_friction"]

    # ── action_rate: standup's ramp, not the roller's ────────────────────────
    # The roller env climbs to -2.0 for a calm gait. That is a motion-blocker: it
    # slows down the fast action a rise from the back needs (standup documents
    # that too strong an action_rate killed that recovery). Smoothness here is
    # carried by joint_torque_rate_l2.
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            # standup's ramp: much gentler than before (-0.4 -> -1.0 from 500).
            # A strong action_rate early is a tax on exploration; -1.0 only
            # arrives at 1500, after the standup skills have settled in.
            "weight_stages": [
                {"step": 0,                        "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV,  "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV,  "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # ── POLISH curricula, introduced LATE ────────────────────────────────────
    # This is the real cause of this env's freeze, and the lesson is borrowed from
    # standup, which established it across two broken runs: "the same weights
    # active from step 0 prevent the flips from ever being DISCOVERED
    # (attempt-tax on exploration)". In other words: ANY tax on attempts during
    # the discovery phase makes "do nothing" win. Refining the gates and lowering
    # the magnitudes changed nothing — "the fix is timing, not magnitude".
    #
    # ground_state_mix finishes ramping the hard poses at iteration 2500; from
    # 3000 the skills exist and ground resets keep exercising them, so these
    # penalties refine execution instead of blocking discovery.
    #
    # ⚠️ If the standup degrades AFTER 3000, soften the LAST stage — do NOT move
    # the introduction earlier.
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "arrival_damping",
            "weight_stages": [
                {"step": 0,                         "weight": 0.0},
                {"step": 3000 * NUM_STEPS_PER_ENV,  "weight": -0.025},
                {"step": 4000 * NUM_STEPS_PER_ENV,  "weight": -0.05},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,                        "weight": 0.0},
                {"step": 3000 * NUM_STEPS_PER_ENV, "weight": -1e-3},
            ],
        },
    )

    # ── Ramped pushes ────────────────────────────────────────────────────────
    # push_robot is inherited from the roller env (±0.2 m/s, every 3–6 s) but
    # without a curriculum. A shove from step 0 disturbs the standup bootstrap, so
    # we ramp it in like standup does.
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0, "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 500 * NUM_STEPS_PER_ENV, "velocity_range": {
                    "x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                {"step": 1000 * NUM_STEPS_PER_ENV, "velocity_range": {
                    "x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
            ],
        },
    )

    return cfg


# ── RL runner config — identical to standup ───────────────────────────────────
MicroduckRollerStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # the normalizer MUST be baked into the ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Symmetry OFF: SYMMETRY_CFG is wired for the old 51D layout and breaks on
        # the 61D one (same situation as every v1.5+ env).
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_standup",
    run_name="roller_standup",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)
