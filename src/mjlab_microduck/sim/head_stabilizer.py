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
                 gain: float = 0.6, damping: float = 1e-3):
        self.alpha = float(alpha)
        self.yaw_tau = float(yaw_tau)
        self.gain = float(gain)
        self.damping = float(damping)
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

        # The camera's home orientation with the trunk's heading divided out. Everything the servo
        # aims at is this, re-planted on the current heading -- so "level" means "as level as it was
        # standing at home", not an arbitrary world axis.
        mujoco.mj_forward(model, data)
        R_cam0 = data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        R_trunk0 = data.xmat[self.trunk_body].reshape(3, 3).copy()
        self.R0 = _rz(-_yaw_of(R_trunk0)) @ R_cam0
        self.yaw_f = _yaw_of(R_trunk0)
        self._jacr = np.zeros((3, model.nv))

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

        w = _log_so3(R_des @ R_cam.T)            # world-frame rotation taking actual -> desired
        if not np.isfinite(w).all():
            return
        mujoco.mj_jacBody(model, data, None, self._jacr, self.cam_body)
        J = self._jacr[:, self.dof_adr]          # 3 x 4, rotation only
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + self.damping * np.eye(3), w)
        target = data.qpos[self.qpos_adr] + self.gain * dq
        data.ctrl[self.act] = np.clip(target, self.lo, self.hi)
