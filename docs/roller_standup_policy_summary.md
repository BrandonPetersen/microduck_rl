# Policy `roller_standup` — getting back up on roller skates

**Goal**: the microduck (on rollers) starts from the ground — face-down or face-up — and gets back **up onto its wheels**, then **holds** the stand.

- **Task**: `Mjlab-RollerStandUp-Flat-MicroDuck`
- **File**: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- **Base**: derived from the roller env (`velocity_rollers`) → same robot, same physics/DR, **same 61D observation** (runtime-swappable, loadable via `--new-cmd-obs`).
- **Spec**: `docs/superpowers/specs/2026-08-04-roller-standup-design.md`
- **Blind policy**: no terrain scan; proprioception + `projected_gravity`.

## Heights (measured, not guessed)

| pose | feet model | rollers model |
|---|---|---|
| standing | 0.1172 → `STAND_Z=0.115` under load | 0.1407 → **`ROLLER_STAND_Z=0.138`** |
| face-down (rest) | 0.075 | 0.075 |
| face-up (rest) | 0.048 | 0.048 |

The ground rest heights are identical on both models: it is the trunk shell that touches, not the feet.

## ⚠️ Joint indices — SERVO-ONLY view (14 joints)

The pose rewards index through `mdp._servo_joint_pos`, which selects
`^(?!passive_).*`: the 14 servos, **without** the wheels or the backlash hinges. The
indices are therefore written in that canonical view, identical to the walker's:

```
0-4   left leg      5-8   neck / head      9-13  right leg
```
`_LEG_JOINTS = [0-4, 9-13]`, `_NECK_JOINTS = [5-8]`. No more `_WHEEL_JOINTS`: the wheels
do not exist in that view, and their DR targets them through the `^passive_.*wheel` regex.

**Do not "fix" these back to full-array positions** (`[0-4, 11-15]`, wheels interleaved at
5,6 and 16,17): that was correct before the `mdp.py` migration, and it has made the env
untrainable ever since. See the regression described further down. Locked on both the rollers
and rollers+backlash models by
`tests/test_roller_standup_cfg.py::test_joint_indices_are_in_the_canonical_servo_space`.

## Reset — ground start

`set_random_ground_state`: face-down (`prone_z` 0.076–0.09, floor raised because the belly only clears the ground from 0.0752) / face-up / **already standing** (`standing_z` 0.134–0.144), ±10° of pitch/roll noise. No "sitting" bucket. The "standing" bucket is necessary: without it the policy learns to rise but not to hold, and falls right back down.

Face-up spawns additionally get **±90° of roll noise** (`face_up_roll_max`) — the built-in reverse curriculum, see the fix section below.

**`ground_state_mix` curriculum** (easy → hard, the back last):

| iter | standing | face-down | face-up |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

## Rewards

**Task block**, aligned on the evolved `standup` recipe (weights at 1/4 of the initial
version; the internal ratios and all the `std` values are unchanged):

| reward | weight | |
|---|---|---|
| `height_stand_l1` | **7.5** | must dominate the block — it is what makes "stay on the ground" net negative |
| `standing_composite` | 3.75 | multiplicative height × upright × pose score, **support-gated** |
| `pose_stand_legs` | 2.0 | target pose = HOME (std 0.5), **support-gated** |
| `upright_linear` / `upright_sharp` | 1.5 / 1.5 | `upright_sharp` height-gated, std 0.3 |
| `pose_stand_l1` | 1.25 | L1 bootstrap |
| `height_stand` / `height_stand_sharp` | 1.0 / 1.0 | std 0.04 (wide) and 0.015 (tight) |
| `com_upward_velocity` | 0.75 | pays for the rise, cut off at `ROLLER_STAND_Z + 0.010` |
| `gentle_rise` | **+0.005** | POSITIVE weight (see the sign bug below); 0.005 = measured ceiling |

**Anti-violence, introduced only at iteration 3000** (see the timing lesson below):
`arrival_damping` (0 → −0.025 → −0.05) and `joint_torque_rate_l2` (0 → −1e-3).

Inherited regularisers: `body_ang_vel` **−0.05** (motion-blocker, keep it LIGHT),
`angular_momentum` −0.02, `action_rate_l2` (base −0.1, gentle ramp to −1.0 **at 1500**),
`neck_joint_pos_l2` −0.5 (head upright), `joint_torques_l2` −1e-3,
`action_over_limit` −0.5, `self_collisions` −1.0.

Dropped: every skating reward, plus `feet_flat` (the blades are not flat during the rise), `hip_roll_neutral` (getting up requires spreading the legs), and `neck_action_rate_l2` (attempt tax, see the fix section).

## The rolling-friction curriculum

The wheels roll, so there is no longitudinal grip to push against the floor. The **rolling-friction curriculum is INVERTED** (the roller env ramps it up, here it comes down):

| iter | frictionloss | |
|---|---|---|
| 0 | 0.05 | wheels nearly locked → the standup works like it would on feet |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | the real rolling value |

⚠️ **Not yet tested.** Run fmt83tri showed no drop at any stage, but that run never stood up (see the results section) — so the measurement cannot speak for a wheel-supported rise. Do not compress this curriculum before a run where the standup actually works.

**Sim2real**: only checkpoints after iter 4000 are deployment candidates. Before that, the policy leans on a friction that does not exist on the real robot.

## Command

`twist` slot neutralised: `lin_vel_x`/`lin_vel_y` ±0.01, `ang_vel_z` **±0.05** (5× wider — same
choice as `standup`). `head_pose` / `body_pose` slots **zero-padded** (roller convention). Target deployment: `--standing` alongside the roller policy in `--walking`, with the automatic switch on command magnitude (`infer_policy.py:262`, threshold 0.05); the twist slot is left at zero there (`infer_policy.py:239`).

**Caveat**: `infer_policy.py` is the local sim/keyboard script. The robot runtime is the Rust `microduck_runtime` binary, absent from this repo — it has not been verified that it exposes a `--standing` equivalent. The crouch handoff doc only lists `--model`, `--ground-pick`, `--fold-policy`. To be confirmed.

## Backlash variant

`Mjlab-RollerStandUp-Flat-Backlash-MicroDuck` — ±1° of gear play in series per servo,
with the firmware PD closing on the encoder **through** the play, like the real servo whose
encoder sits on the gearbox output. Model `MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG`
(wheels + backlash, 32 joints of which 18 passive). Obs and actions stay at 14 joints, so
runtime and export are unchanged.

```bash
uv run train Mjlab-RollerStandUp-Flat-Backlash-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
```

Rewards, weights, ground reset and curricula are **identical** to the base task
(checked by `test_backlash_variant_keeps_the_recovery_recipe`): only the model and how joints
are read change.

⚠️ This variant is **only safe since the move to servo-only indices**. With the old indices
written against the full array, it would have silently rewarded wheels and backlash hinges,
without crashing.

**Why it is worth the detour here**: the standup pushes against the floor with compliant
servos, and gear play is exactly the kind of sim2real gap that makes something passing in sim
fail for real — the symptom observed on the back. Compare against the base task to settle it.

## Terminations

`fell_over` **removed** (the robot starts fallen). `nan_state` inherited. `nan_policy="sanitize"` on the actor/critic obs.

## Network / PPO

Actor and critic `(512, 256, 128)` elu, `obs_normalization=True`. PPO `lr=1e-3` adaptive, `desired_kl=0.01`, `gamma=0.99`, `lam=0.95`, `num_steps_per_env=24`, 6 s episode, `max_iterations=15000`. **Symmetry OFF** (`SYMMETRY_CFG` is wired for the 51D layout).

## Commands

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 6000
uv run scripts/play_latest.py        # alias md-play
uv run --with pytest pytest tests/test_roller_standup_cfg.py tests/test_wheel_support_gate.py -q
```

### ⚠️ Seeing back starts at play time

A play **never** shows a back start by default: the play env is rebuilt from scratch, so
`common_step_counter` restarts at 0 and the curriculum applies its stage 0, where
`face_up_prob = 0`. You only ever see 50 % face-down / 50 % standing, whatever the maturity
of the loaded checkpoint. Yet the back is the hardest case, the one worth inspecting.

`STANDUP_PLAY_FACE_UP` forces the mix (same pattern as `SLOPE_PLAY_DIFFICULTY` in
`roller_slope`), **on the `play=True` path only** — training and its easy → hard curriculum
are untouched:

```bash
STANDUP_PLAY_FACE_UP=1.0 md-play    # 100 % back starts
STANDUP_PLAY_FACE_UP=0.4 md-play    # the final curriculum stage's mix
STANDUP_PLAY_FACE_UP=none md-play   # default (stage 0, no back start)
```

The remainder (`1 - face_up`) is split face-down:standing in the last stage's 2:1 ratio, so
`0.4` reproduces the end-of-training mix exactly (0.40 / 0.20 / 0.40).

⚠️ `num_envs` defaults to **1**, so a single duck shows one pose at a time (redrawn every
6 s). Pass `--num-envs 15` to see the mix side by side: at 0.4 that is 6 back / 6 face-down /
3 standing.

## 🔧 History of failures and of the resynchronisation

### Sign bug: `gentle_rise` was rewarding violence

**Symptoms** (checkpoint 4000+): very abrupt motion, the head banging the floor,
back recovery failing. **Present in sim too** → neither sim2real nor a young checkpoint.

`trunk_vertical_accel_penalty` already returns `-|a_z|` (`mdp.py`); multiplied by the
**−0.02** weight inherited from `standup`, that gave `+0.02·|a_z|` — **the more brutally the
trunk accelerated, the more the policy was paid**. Confirmed: `Episode_Reward/gentle_rise = +0.0118`
on run `vweolw91`, the only penalty term logging positive.

`mdp.py` mixes two sign conventions:

| term | what the function returns | correct weight |
|---|---|---|
| `height_stand_l1`, `pose_stand_l1`, `gentle_rise` | `-abs(...)`, already negative | **positive** |
| `joint_torques_l2`, `joint_torque_rate_l2`, `action_rate_l2`, `body_impact_cost` | positive magnitude | **negative** |

Locked by `test_already_negative_penalties_use_positive_weights`. The same bug existed
in `standup` and `sitstand` (run `7ev90yd9`); **both have been fixed since**.

### Failure #1: head-impact penalty → frozen policy

Tried with `velstand`'s values (−1.0, threshold 2.0): **the policy converged to lying down,
inert.** Measured (run `d8rnko6p`): `head_impact_penalty` −1.01/step, the largest negative
term, `standing_composite` collapsed from +14.3 to +3.3.

**The lazy optimum that makes such a freeze possible**: `pose_stand_legs` stayed at **+7.72 of 8**
while the robot was lying flat — the legs sit at HOME in a lying pose, so the reward is
collected almost for free. `height_stand_l1` is the term that counterbalances that (locked by
`test_height_l1_stays_the_dominant_task_term`), and the support gate now handles it directly.

### The real lesson: it is TIMING, not magnitude

`standup` established the general law across two broken runs: *"the same weights active from
step 0 prevent the flips from ever being DISCOVERED (attempt-tax on exploration)"*, and
*"the fix is timing, not magnitude"*. Any tax on attempts during the discovery phase makes
"do nothing" win. Both of this env's additions (`head_impact_penalty` at −1.0 **and**
`joint_torque_rate_l2` at −2.0) were active from step 0.

### Recipe resynchronised on `standup`

| | before | now |
|---|---|---|
| whole task block | weights ×4 | **÷4** (`standing_composite` 3.75, `pose_stand_legs` 2.0, `height_stand_l1` 7.5…) |
| `gentle_rise` | −0.02 (a reward) | **+0.005** — measured ceiling: 0.01 contributed to the freeze |
| `com_upward_velocity` | 3.0 | **0.75** |
| `arrival_damping` | absent | **`body_ang_vel_at_height`**, height+tilt gated, 0 → −0.025 at 3000 → −0.05 at 4000 |
| `joint_torque_rate_l2` | −0.2 from step 0 | **0** → −1e-3 at 3000 |
| `action_rate_l2` ramp | −0.4 → −1.0 from 500 | **−0.1 → −1.0 at 1500** |
| `head_impact_penalty` | tested at −1.0 | **absent** |

Dividing the task rather than raising the dampers fixes the task/damper ratio
(measured at ~35:1, now ~9:1) **without** turning a damper into a motion-blocker.

`arrival_damping` targets the real failure loop — rise → overshoot vertical → tip → retry.
Its height gates are transposed onto `ROLLER_STAND_Z` (0.113 / 0.133) and **not** copied from
the walker (0.09 / 0.11), which would open the gate while the roller robot is still 3 cm below
its stand, i.e. mid-rise. The tilt gate is essential: without it, the final straightening of a
folded rise is itself a large rotation, and taxing it raises a wall just before arrival.

⚠️ **If the standup degrades after 3000, soften the LAST stage — do not move the introduction
earlier.**

### 🐛 Regression: out-of-bounds joint indices

The `mdp.py` migration to `_servo_joint_pos` made the env **untrainable** with nothing
signalling it. `joint_indices` is now interpreted in the **14-joint servo-only view**
(`^(?!passive_).*`, so without wheels or backlash); the constants targeted the 18-joint full
array, so indices 14 and 15 fell out of bounds → `index out of bounds` on GPU.

**The 37 config tests passed anyway**: they build `cfg` without ever calling the rewards.
Only a real run reveals it — that is the structural limit of those tests, and 3 real
iterations must be launched after any index or sensor change.

Benefit: the servo-only view is **identical** on the rollers model and on rollers+backlash
(32 joints, 18 passive), so the indices are now backlash-proof — checked on both models by
`test_joint_indices_are_in_the_canonical_servo_space`.

### Method lesson

The first three fixes were applied at once, so the freeze could not be attributed with
certainty. One fix at a time.

## Out of scope

Folding the standup into the skating policy (`velstand` recipe); side-start buckets; a rough variant; trunk/head impact penalties.

No reward penalises horizontal trunk velocity (`root_link_lin_vel_w[:, :2]`): "getting up while rolling far away" is an unpenalised outcome that scores in full. A deliberate decision (not an oversight): a stillness reward that was not height-gated would also penalise the translation that getting up from the ground physically requires — the "motion-blocker" failure mode `standup` documents. Candidate if the problem is confirmed: a height-gated stillness (near `ROLLER_STAND_Z` only).

---

## 🔴 Run `fmt83tri` — the head tripod, and what the port had lost

First run of the resynchronised recipe (4096 envs, 6000 iters). **Failed, diagnosed.**

### What the policy was doing

A **head tripod**: head planted on the floor, trunk levered up to standing height at ~55° of
tilt, legs left at HOME. Confirmed by eye on `model_500` as well as `model_3500` — **there was
never a standup**, at any stage.

Why it paid, measured per step at iteration 3625:

| term | value | max | % |
|---|---|---|---|
| `pose_stand_legs` | 1.991 | 2.00 | **99.5 %** (flat since iter 250) |
| `height_stand_sharp` | 0.505 | 1.00 | 50 % |
| `upright_linear` | 0.852 | 1.50 | 57 % → cos(tilt) 0.57 → **55°** |
| `standing_composite` | 0.915 | 3.75 | 24 % |
| **total positive** | **5.16** | **10.75** | **48 %** |

Keeping 48 % of the stack without ever standing up = AGENTS.md's audit failing. The
multiplicative composite did collapse on the uprightness factor, but it only weighs 3.75 of a
10.75 positive mass: **a multiplicative score does not break a compromise that the rest of the
stack finances.**

### The tell in the curves

`standing_composite / max` tracked `standing_prob` to within +0.05 at every stage
(0.50→0.59, 0.35→0.40, 0.25→0.27, 0.20→0.24). In other words **all the standing reward came
from envs spawned already standing**. The three "drops" at 625 / 1625 / 2625 were not a skill
degrading: they were the only paying bucket shrinking.

⚠️ The walker `standup` describes **exactly this signature** for its two broken runs of
2026-07-24: *"standing metrics drop at the ground_state_mix stages instead of recovering like
the reference run"*. This env had reproduced the walker's broken runs, not its working one.

### Fix 1 — support gate (`wheel_support_gate`)

`standing_composite` and `pose_stand_legs` go through their `*_on_wheels` variants:
multiplied by a binary gate that is 1 **only if the robot is carried by its wheels alone**
(one tire is enough; head, trunk or limbs on the floor close it). A gate, not a penalty: a
penalty gets negotiated ("the head costs me 1.0 but earns me 2.9"), zero times anything does
not.

**The CLIMB shaping stays ungated** (`height_stand`, `height_stand_l1`,
`upright_linear`) — otherwise nothing pulls the robot off the floor any more. Locked by
`test_climb_shaping_stays_ungated`.

New sensors: `head_ground_contact` (`jaw_soft`), `trunk_ground_contact`
(`trunk_base`, **`mode="body"` and not `"subtree"`** — the subtree contains the tires, so the
gate would stay closed while standing) and `limbs_ground_contact` (see the leak below).

### Fix 2 — `neck_action_rate_l2` dropped

After the gate, the policy **froze** from face-down (observed at iter 500). That was not a
regression: the gate had removed the tripod, the only behaviour the policy had ever found, and
there was nothing else within exploration reach.

The arithmetic it then faced, from face-down: **stay still ≈ −0.39/step, move ≈ −4.2/step.**
Doing nothing won by a factor of 10.

`neck_action_rate_l2` (−0.5) was the biggest item: measured **−1.359/step**, the largest term
in the whole reward, ~1.9× the positive task block. It **double-taxes** the 4 head joints,
already covered by `action_rate_l2` (effective weight 0.6 against 0.1 per leg joint at stage 0).
It arrived by inheritance from the **skating** recipe and had never been audited for a standup.
The walker `standup` drops it explicitly (`microduck_standup_env_cfg.py:485`).

⚠️ **The two neck terms pull in opposite directions.** `neck_joint_pos_l2` (−0.5, head far from
neutral) fights the tripod → **kept**. `action_over_limit` (−0.5) is kept too: it penalises
commands outside `ctrlrange`, not motion. Locked by
`test_neck_action_rate_is_dropped_but_neck_position_is_kept`.

### Fix 3 — `face_up_roll_max = 90°`

**The parameter was missing**, so it defaulted to 0: every back start was perfectly flat. Yet
the walker documents that case as hopeless — *"back-recovery was seed-lucky
(1 success / 3 failures) because the reward landscape from flat supine to prone is FLAT"*.

Measured here on 256 spawns: **tilt reads 90.0° for all of them, whatever the roll** — rolling
about the long axis does not change the distance from vertical, so `upright_linear` stays at
≈ 0 throughout the gesture. Direct confirmation that no gradient guides the roll.

The roll noise makes a fraction of episodes start **part-way through the roll**: a built-in
reverse curriculum with no stage to tune. Verified with no spawn penetration (0/256 below the
floor); 42 mm median drop at reset, a known artefact of the `prone_z` floor shared between
back and belly.

### ⚠️ The friction curriculum: measurement retracted, not confirmed

At all three stages (1000 / 2000 / 3000, i.e. 0.05 → 0.003, a factor of 17),
`standing_composite` **did not drop** — at 2000 and 3000 it even rose just after. The two
curricula land on different iterations (600/1500/2500 vs 1000/2000/3000), so the attribution
would have been clean.

**But this was first written up as "the wheels are NOT the hard part", and that conclusion was
premature.** Run fmt83tri never stood up: its composite tracked `standing_prob` and its
behaviour was a head tripod, which does not carry the body on the wheels at all. A friction
that only matters to a *wheel-supported* rise could not possibly have shown up in it.

The first real test is a run where the face-down standup works — stages **3000 (0.003)** and
**4000 (0.0015)** are then the ones to watch. **Do not compress this curriculum before that
evidence exists.**

Method note worth keeping: a null result measured on a degenerate run is not a null result.
The question has to be *reachable* by the behaviour being observed.

### 🐛 The v1 gate's leak — hips and shins

First gated run: `pose_stand_legs / 1.9 = 0.67` while `standing_prob = 0.50`, misread as
"a third of the face-down starts get up". **The video said otherwise**: robot sprawled,
motionless.

Cause: the rollers model carries only **12 collision geoms** —

```
trunk_base  np_f970 (battery, at the rear)      hip_l, hip_l_2   hip
jaw_soft    top_head_shell, jaw, bottom_head    leg, leg_2       leg
tire ×4
```

**The trunk shells are VISUAL-only.** A robot sprawled on its hips and shins, one tire grazing
the floor and the head held up, therefore triggered neither `head_ground_contact` nor
`trunk_ground_contact`: the gate opened while lying flat on the ground.

Fix: a third sensor `limbs_ground_contact` (`^(hip_l|hip_l_2|leg|leg_2)$`). "Gate open" now
means exactly **"only the tires touch the floor"**.

Verified in BOTH directions, which the first fix had not done:

| | gate |
|---|---|
| 128 envs standing on wheels, HOME ctrl, steps 0–25 | **0.99 → 1.00** (`limbs` never triggered) |
| the same after toppling (step 100, head down) | 0.39 |
| sprawled on hips + shins (unit test) | **0** |

Smoke: `pose_stand_legs` 0.0896 → 0.0288 on a random policy — the leak was worth 3×.

⚠️ **Method lesson.** A contact-based gate reads "this part is not touching", not "the robot is
standing". Any incomplete list of forbidden contacts is a silent leak that *looks like progress
in the curves*. The mandatory counter-test is twofold: does the gate open while standing, AND
does it stay closed on every stable sprawl?

### Reading the metrics — what is trustworthy

| gauge | computation | what it says |
|---|---|---|
| `standing_composite / 3.75` **vs** `Curriculum/ground_state_mix` | ratio | **the main test.** The latter logs `standing_prob`. Equality = only the already-standing envs score = no standup at all |
| `pose_stand_legs / 1.9` | ≈ fraction of time with the gate open | honest **only since the leak fix** |
| `\|height_stand_l1\| / 7.5` | height error (m) | 32 mm on the tripod, 4 mm when it holds |
| `acos(upright_linear / 1.5)` | trunk tilt | **55° = tripod**, 46° = propped on the hips, < 26° = standing |

⚠️ `standing_composite` crushes through an uprightness factor of std 0.40: at 32° of tilt that
factor is already 0.39. A low composite can therefore mean "leaning" rather than "not
standing" — `pose_stand_legs` (gated) is what settles it.

### ✅ `ROLLER_STAND_Z = 0.138` — verified under load on the rollers model

The unverified assumption was the sag: `0.1407` did come from exact kinematics on
`scene_rollers.xml`, but the step to `0.138` **borrowed** the ~2 mm of sag measured on the
model WITHOUT wheels.

Direct measurement (512 envs spawned standing, HOME ctrl, DR off, envs at tilt < 5° only):

| step | upright envs | median z | offset from 0.138 |
|---|---|---|---|
| 5 | 512 | 0.1393 | +1.3 mm |
| 10 | 512 | 0.1388 | +0.8 mm |
| 20 | 12 | 0.1386 | +0.6 mm |

Real sag: 0.1407 → 0.1386 = **2.1 mm**, against 2.2 mm borrowed. The borrow was right.
`height_stand_sharp` having a 15 mm std, a 1 mm target error costs it 0.4 % — no effect.

⚠️ **Measured side finding, more important than the height itself**: 512 envs upright at step
10, only **12 left at step 20**. With no DR, no tilt noise, no push. Standing on four free
wheels with a PD toward HOME, the robot topples in **0.4 s**.

The "already standing" bucket is therefore NOT a free ride: holding the stand on rollers is
already an active control problem. Consequence for reading the curves: envs spawned standing do
not automatically score `standing_prob`, they only score if they hold — part of the gap between
`standing_composite/3.75` and `ground_state_mix` comes from that, not only from the ground envs.

---

## 🟡 Iteration 2500: face-down works, the back never moves

First observation of a real standup on this env — **from face-down only**. From the back the
robot stays lying, motionless.

### Measured: the roll has no gradient, and its first half is downhill

Reward landscape swept along the supine→prone roll (10° steps, 8 envs per angle, no policy —
this is a property of the env, not of a checkpoint):

| roll | total | `height_stand_l1` | `upright_linear` | gated terms |
|---|---|---|---|---|
| **0° (flat on back)** | −0.719 | −0.686 | −0.006 | **0** |
| 50° | −0.915 | −0.731 | +0.029 | 0 |
| **90° (on the side)** | **−0.757** | **−0.739** | +0.032 | **0** |
| 120° | −0.782 | −0.725 | +0.069 | 0 |
| **180° (face down)** | −0.563 | −0.628 | +0.092 | 0 |

1. **The whole rollover is worth +0.156/step.** Against `action_rate_l2` at its full −1.0
   (≈ −0.2/step for a smooth policy), moving is net negative.
2. **The first half is DOWNHILL**: on the side the trunk sits 7 mm lower than on the back
   (z 0.0465 → 0.0395), so `height_stand_l1` actively penalises starting the move.
3. **Every gated term reads 0.0000 across the full 180°** — by design, but it means the only
   live signal during the entire gesture is `height_stand_l1`, pointing the wrong way.

This also explains why `face_up_roll_max` alone did not unlock the back: **a reverse curriculum
supplies on-policy DATA, it does not create a gradient**, and the data said rolling does not pay.

### Fix — `height_progress`, potential-based Δz (weight 200)

Ported from `velstand` (same function, weight 30 there). Pays `Δ min(z, ROLLER_STAND_Z)`:
rising pays, **holding pays exactly zero**, falling refunds, hopping above the stand pays
nothing extra. Ungated on purpose — the floor is where the signal is needed.

**Why a Δ and not a wider `height_stand` Gaussian.** Widening the std 0.04 → 0.08 buys the same
floor gradient (0.006 → 0.077 per cm, ×13) but pays **0.267/step for merely lying on the back**
(against 0.005 today) while the standing robot gains nothing — the standing/lying gap shrinks
and a free floor reappears. Any reward that pays for *being* at a height pays for being there
effortlessly. That is the exact mechanism behind the tripod.

**Why it prescribes no technique.** It measures trunk height, not posture. Roll, pike, pivot on
a shoulder, or something nobody has pictured — all paid identically, per centimetre gained. The
path stays what RL is supposed to discover. (An earlier proposal to reward *turning onto the
belly* was rejected for exactly this reason: it would have imposed the rollover and paid zero
to any other solution.)

Being potential-based (Ng et al.) it is **policy-invariant** — it cannot create a new optimum,
which is what makes a large weight safe.

**Weight derivation** (keep it checkable, see `test_height_progress_weight_cancels_the_action_tax`):
full rise 0.046 → 0.138 = 0.092 m → total **+18.4**; a 2 s climb (100 steps, 0.92 mm/step) pays
**+0.18/step**, cancelling `action_rate_l2`'s ≈ −0.2/step. `velstand` uses 30 because there it
is a last-mile helper on a working recovery; here it is the primary gradient across 9 cm of
unpaid ground.

⚠️ Its weight multiplies a **Δz in metres** (~0.001/step), not a level in [0, 1] — 200 here is
not comparable to `height_stand_l1`'s 7.5. Do not "fix" them into agreement.

### Verified in the env (not on paper)

```
weight=200  ceiling=0.138
hold 40 steps          : -0.043      (claim: ~0)                     ✓
forced +1 cm           : +2.000 exactly, every cm                    ✓
above the ceiling      : +0.000                                      ✓
reset settle           : -1.82 once, bounded and action-independent  ✓
```

The `fresh` guard (`episode_length_buf <= 1`) absorbs most of the reset drop: of the 41.7 mm
fall, only 9.1 mm is charged.

### Reading it

⚠️ **`height_progress` is the one term that legitimately logs either sign** — it is a signed Δ,
not a penalty. It does not belong in the "every penalty ≤ 0" check.

Random policy reads ≈ **−0.03** (flailing loses height). A population that rises and holds
should read ≈ **+0.065/step** (+18.4 spread over a 300-step episode). **Crossing zero is the
signal that the back is being learned.**
