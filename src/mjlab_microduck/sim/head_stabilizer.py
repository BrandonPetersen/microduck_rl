"""A horizon-lock gimbal on the duck's 4-DOF neck, for SLAM experiments.

WHY THIS EXISTS. duckslam's validation ladder (microduck_vslam, docs/validation) found that
vision-only odometry fails on this robot not because of depth quality or RTAB-Map tuning, but
because the gait swings the camera far more than it translates it: ~3.46 deg of rotation against
~10.7 mm of travel per frame, which at the arena's ~0.94 m scene depth is 26 px of rotational
optical flow burying 5 px of parallax. Translation is what visual odometry must recover, and
parallax is the only thing that carries it.

Removing that rotation from already-captured frames (by warping them, which is exact for a rotation
about the optical centre) took vision-only ATE from 249 cm to 15 cm. But a warp cannot answer the
two questions that decide whether this is worth building:

  * does the WALK survive the head moving this much? The neck carries mass; counter-rotating it
    every step changes the angular momentum the balance controller is working against.
  * can the NECK ACTUATORS actually track it? They are position servos with kp=0.55 and a
    +-0.96 Nm force range, asked here for ~35 deg/s sustained and ~80 deg/s peaks.

Both are physics questions, so this drives the real actuators rather than teleporting the head:
`data.ctrl` for the four neck joints is overwritten, the position servos do the work, and the
reaction torques land on the trunk exactly as they would on hardware. Setting `qpos` directly would
have been easier and would have silently destroyed both answers.

WHAT IT HOLDS. Pitch and roll level in the world frame, yaw following the trunk's heading through a
first-order filter, with the camera's home tilt preserved. That is what a 3-axis camera gimbal in
horizon-lock does. It deliberately does NOT hold the head fixed in space like a bird: the camera's
TRANSLATION is the parallax signal, and removing it would take away the very thing the estimator is
starving for. Rotation-only is not a concession to having four DOF -- it is the optimum.

The target is anchored to trunk heading rather than to the camera's own past, so it cannot drift:
a servo that chases a filtered version of its own output would slowly lock to wherever it started
and stop following the robot around the room.

`alpha` scales the correction (0 = off, 1 = full), matching the sweep parameter in
microduck_vslam's docs/validation/data/08-stabilize.py so the two can be compared. That sweep found
the benefit peaks near alpha=0.5 and that over-stabilising slightly hurts.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Wire order (`JOINT_NAMES` in body_server) for the neck chain, trunk outward.
NECK_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle, radians)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(th)) * th


def _exp_so3(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_of(R: np.ndarray) -> float:
    """Heading: the world-z rotation that best describes R, taken from its x axis."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


class HeadStabilizer:
    """Resolved-rate orientation servo driving the four neck actuators.

    One `step()` per `mj_step`, so it runs at the physics rate (200 Hz at the 5 ms timestep) rather
    than the daemon's 50 Hz. That matters: the thing being rejected is gait-frequency motion, and a
    controller sampling near that frequency would chase its own tail. It also makes this causal by
    construction -- it reads only the current state, never a future one, which the offline warp
    experiments could not claim.
    """

    def __init__(self, model, data, body, *, alpha: float = 0.5, yaw_tau: float = 0.5,
                 damping: float = 1e-3, iters: int = 3, max_dev_deg: float = 20.0):
        self.alpha = float(alpha)
        self.yaw_tau = float(yaw_tau)
        self.damping = float(damping)
        self.iters = int(iters)
        self.max_dev = np.radians(float(max_dev_deg))
        self.cam_id = body.cam_id
        self.cam_body = int(model.cam_bodyid[self.cam_id])
        # The trunk is the root of this duck's kinematic tree (the body carrying the free joint).
        self.trunk_body = int(model.body_rootid[self.cam_body])

        # The neck's slots within this duck's own actuator/qpos bookkeeping.
        from .body_server import JOINT_NAMES
        wire = {JOINT_NAMES.index(n) for n in NECK_JOINTS}
        slots = [s for s, w in enumerate(body.to_wire) if w in wire]
        if len(slots) != len(NECK_JOINTS):
            raise RuntimeError(f"expected {len(NECK_JOINTS)} neck joints, found {len(slots)}")
        self.act = np.array([body.actuators[s] for s in slots])
        self.qpos_adr = np.array([body.qpos_adr[s] for s in slots])
        self.dof_adr = np.array([body.qvel_adr[s] for s in slots])
        jnt = np.array([int(model.actuator_trnid[a, 0]) for a in self.act])
        self.lo = model.jnt_range[jnt, 0].copy()
        self.hi = model.jnt_range[jnt, 1].copy()

        self.trunk_qpos = int(body.trunk)
        mujoco.mj_forward(model, data)
        R_trunk0 = data.xmat[self.trunk_body].reshape(3, 3).copy()
        self.yaw_f = _yaw_of(R_trunk0)
        self.q_home = data.qpos[self.qpos_adr].copy()

        # The reference orientation is computed for a LEVEL trunk with the neck at home, on a
        # scratch copy -- NOT read from whatever pose the duck happens to be in right now.
        #
        # Reading it live was a real bug with a real cost: `duck-sim` starts at the SIT keyframe, so
        # the reference became "the camera as it points while sitting". The gimbal then spent the
        # whole run trying to hold that orientation, fighting the stand-up, and drove the neck into
        # its stops -- head_pitch pinned at +90 deg, head_yaw swinging 320 deg, roll at both limits.
        # The duck could not stand at all for 250 s while the policy commanded it to walk. A
        # reference derived from geometry cannot drift with the startup pose that way.
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        scratch.qpos[self.trunk_qpos + 3: self.trunk_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        scratch.qpos[self.qpos_adr] = self.q_home
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_camlight(model, scratch)
        self.R0 = scratch.cam_xmat[self.cam_id].reshape(3, 3).copy()
        self._jacr = np.zeros((3, model.nv))
        # A scratch MjData so the solve can evaluate forward kinematics for a CANDIDATE neck pose
        # without disturbing the live simulation.
        self._scratch = mujoco.MjData(model)

        # Two failure modes are designed out here, both measured before they were:
        #  * the neck is REDUNDANT (neck_pitch and head_pitch rotate the camera about exactly
        #    opposite axes), so any scheme that INTEGRATES joint increments walks the pair up the
        #    null direction until a joint saturates. Measured: neck_pitch pinned at its +60 deg
        #    limit within 1.5 s, camera 20 deg off level -- worse than no gimbal.
        #  * an integrator on top of a slow, weak plant (kp=0.55) rings. Measured: 80 deg of neck
        #    travel and a camera 52 deg off level.
        # Solving for the ABSOLUTE joint angles from `q_home` every step has neither failure mode:
        # there is no accumulated state to drift or oscillate, and the position actuators do the
        # tracking, so what is left over is the actuator's real limitation rather than the
        # controller's.

    def step(self, model, data, dt: float) -> None:
        R_cam = data.cam_xmat[self.cam_id].reshape(3, 3)
        R_trunk = data.xmat[self.trunk_body].reshape(3, 3)

        # Heading, low-passed: follow where the robot is going, ignore how the gait wags it there.
        yaw = _yaw_of(R_trunk)
        k = min(dt / max(self.yaw_tau, 1e-6), 1.0)
        self.yaw_f += np.arctan2(np.sin(yaw - self.yaw_f), np.cos(yaw - self.yaw_f)) * k
        R_des_full = _rz(self.yaw_f) @ self.R0
        # alpha scales how much of the correction is applied, same convention as the offline sweep
        R_des = R_cam @ _exp_so3(self.alpha * _log_so3(R_cam.T @ R_des_full))

        # Solve, from home, for the neck pose that puts the camera on target given where the trunk
        # is NOW. A few damped-least-squares iterations on a scratch copy: the chain is short and
        # the correction small, so this converges in two or three.
        self._scratch.qpos[:] = data.qpos
        q = self.q_home.copy()
        for _ in range(self.iters):
            self._scratch.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(model, self._scratch)
            # mj_jacBody reads `cdof`, which mj_kinematics does not fill -- without this the
            # Jacobian comes back zero, the correction collapses to nothing, and the gimbal
            # silently does nothing at all while appearing to run.
            mujoco.mj_comPos(model, self._scratch)
            mujoco.mj_camlight(model, self._scratch)
            R_try = self._scratch.cam_xmat[self.cam_id].reshape(3, 3)
            e = _log_so3(R_des @ R_try.T)
            if np.linalg.norm(e) < 1e-4:
                break
            mujoco.mj_jacBody(model, self._scratch, None, self._jacr, self.cam_body)
            J = self._jacr[:, self.dof_adr]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(3), e),
                        self.lo, self.hi)
        # Never let the gimbal contort the neck. The measured demand is ~+-5 deg of travel
        # (docs/validation/data/08-neck-demand.txt), so a 20 deg envelope is generous -- and it is
        # the difference between a gimbal that gives up on a pose it cannot reach and one that
        # drives into the stops and stops the robot standing, which is what the unclamped version
        # actually did.
        q = np.clip(q, np.maximum(self.q_home - self.max_dev, self.lo),
                    np.minimum(self.q_home + self.max_dev, self.hi))
        data.ctrl[self.act] = q
