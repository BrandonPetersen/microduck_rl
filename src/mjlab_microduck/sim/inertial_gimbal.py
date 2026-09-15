"""Hold the camera's pitch and roll against gravity, not against the head.

WHY A SECOND CONTROLLER. `camera_gimbal.CameraGimbalStabilizer` targets a low-passed copy of the
HEAD's own orientation. That makes it a smoothing filter: everything faster than `tau` is removed,
everything slower is reproduced faithfully. Measured, it leaves 15 deg/s of residual rotation -- and
the part it passes is the slow part, which is invisible on video and is exactly what integrates
into trajectory error. duckslam's rung 18 measured the consequence: 0.895 deg of residual rotation
per frame, but 5.74 deg of ACCUMULATED heading error, and a rotation/parallax ratio of 1.4 where
visual odometry wants <= 0.3.

WHAT CHANGES. Only the reference. The tracking loop -- absolute IK solved from home every step, no
accumulated state -- is unchanged, because that formulation is what fixed the null-space drift and
the actuator ringing and neither should be re-litigated.

    old:  R_des = lowpass(head orientation, tau=0.5 s)
    new:  R_des = Rz(yaw washout) . lowpass(tilt RELATIVE TO GRAVITY, tau=5 s)

Pitch and roll become absolute. A trunk that pitches over two seconds moves the old reference with
it; gravity says the horizon did not move, so the new one rejects it.

WHY PITCH AND ROLL BUT NOT YAW. This is not a simplification, it is what the physics and the sensor
agree on. Gravity observes roll and pitch driftlessly and says nothing whatever about yaw, and
those are precisely the two axes a walking biped never accumulates -- so they can be held
indefinitely inside the +-30 deg joint range. Yaw has to follow, because nothing observes it
driftlessly without a magnetometer, and because the robot genuinely turns. Every real camera gimbal
is built this way. It also happens to be where the error is: rung 18 measured 19.08 deg of tilt
against 2.66 deg of yaw.

NOT LEVELLING. The aim setpoint tracks the head's mean pitch rather than pinning the camera to the
horizon. Levelling was tried and rejected on measurement: it discarded the robot's ~6.2 deg
downward aim, pushed median scene depth 0.96 -> 1.12 m and cost 17% of the parallax -- and parallax
is the entire signal this exists to protect. The tracking time constant (~5 s) is far slower than
the 1-2 Hz gait, so gait is still fully rejected while a deliberate look-up is not fought.

WHAT IT READS, AND WHY THAT MATTERS. Only two things, both of which the robot already reports in
`body_server`'s `imu` block, plus joint encoders:

    gravity in the trunk frame   -- an accelerometer, low-passed. Gives roll and pitch.
    angular rate in trunk frame  -- a gyro. Gives the yaw rate to hold against.

It deliberately does NOT read the trunk's absolute world orientation, which no robot can measure.
The world-frame conversion at the end is a change of coordinates for MuJoCo's world-frame Jacobian
and cancels out of the solution; the control law's information content is IMU + encoders, so this
controller ports to hardware unchanged.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .camera_gimbal import CAMERA, GIMBAL_JOINTS

# Far slower than the 1-2 Hz gait, so gait is rejected outright, and fast enough that a deliberate
# change of aim is followed within a few steps rather than fought.
TAU_AIM = 5.0
# Yaw washout. Short of this the gimbal holds heading inertially; beyond it, it gives way to wherever
# the robot has actually turned. Too short and gait yaw survives; too long and the joint saturates
# in a sustained turn.
TAU_YAW = 4.0
# Lead compensation for actuator lag. The joints reach their commanded angle roughly 14 ms late,
# which at 2 Hz is ~2 deg of uncancelled trunk motion -- measured as the ONLY band where this
# controller lost to the follow one, with commanded and achieved swing otherwise near-identical and
# no saturation, i.e. a phase error and not an authority one. Commanding where the solution is
# heading rather than where it is cancels it. This is what a real gimbal's rate loop does.
TAU_FF = 0.014


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _log_so3(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(th)) * th


def _exp_so3(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    K = _skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def heading_of(A, fallback=0.0):
    """Heading about gravity of a camera whose orientation in the gravity-aligned frame is `A`.

    Taken from where the camera LOOKS, not from the matrix's first column. A MuJoCo camera views
    along its own -z, so in this frame a level camera is a 90 deg rotation whose A[0,0] and A[1,0]
    are both exactly zero -- `atan2(0, 0)` is undefined there, and numerical noise then swings the
    extracted heading arbitrarily. Measured, that made the controller AMPLIFY an imposed tilt
    (19.7 deg out of 6.0 deg in) rather than reject it.
    """
    a = A @ np.array([0.0, 0.0, -1.0])
    if abs(a[0]) < 1e-6 and abs(a[1]) < 1e-6:      # looking straight up or down: heading undefined
        return float(fallback)
    return float(np.arctan2(a[1], a[0]))


def gravity_aligned_frame(g_trunk):
    """Trunk <- gravity-aligned frame: z is up, x is the trunk's forward flattened onto the horizon.

    Everything the controller reasons about lives here, because in this frame "hold pitch and roll"
    is simply "keep the tilt part constant" and needs no Euler convention to state.
    """
    u = np.asarray(g_trunk, float)
    n = np.linalg.norm(u)
    u = np.array([0.0, 0.0, 1.0]) if n < 1e-9 else -u / n      # gravity points down; up is -g
    fwd = np.array([1.0, 0.0, 0.0])
    x = fwd - np.dot(fwd, u) * u
    if np.linalg.norm(x) < 1e-6:                                # trunk nose-up; any horizontal axis
        x = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], u) * u
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(u, x), u], axis=1)


class InertialGimbalStabilizer:
    """Camera gimbal whose pitch and roll are held against gravity and whose yaw washes out.

    Drop-in for `CameraGimbalStabilizer`: same joints, same actuators, same absolute-IK tracking.
    The two differ only in what they aim at, which is the whole point -- they can be run on the same
    capture and compared directly.
    """

    def __init__(self, model, data, *, prefix: str = "", alpha: float = 1.0,
                 damping: float = 1e-4, iters: int = 4,
                 tau_aim: float = TAU_AIM, tau_yaw: float = TAU_YAW,
                 tau_ff: float = TAU_FF):
        self.alpha, self.damping, self.iters = float(alpha), float(damping), int(iters)
        self.tau_aim, self.tau_yaw = float(tau_aim), float(tau_yaw)
        self.tau_ff = float(tau_ff)
        self._q_prev = None

        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, prefix + CAMERA)
        if self.cam_id < 0:
            raise RuntimeError(f"no camera {prefix + CAMERA!r}; was add_camera_gimbal() applied?")
        self.cam_body = int(model.cam_bodyid[self.cam_id])
        self.head_body = int(model.body_parentid[self.cam_body])
        self.trunk_body = int(model.body_rootid[self.cam_body])
        # The trunk's free joint supplies the gyro, exactly as body_server reads it for `imu`.
        self.trunk_dof = int(model.body_dofadr[self.trunk_body])

        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)
                for j in GIMBAL_JOINTS]
        self.act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + j)
                             for j in GIMBAL_JOINTS])
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0].copy()
        self.hi = model.jnt_range[jids, 1].copy()
        self.q_home = np.zeros(len(jids))

        # The camera's fixed mount inside the head, from geometry at zero gimbal -- taken from a
        # scratch model at qpos0 rather than from whatever pose the duck happens to be in, because
        # reading it live is what once pinned a stabiliser to the SIT keyframe.
        mujoco.mj_forward(model, data)
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = model.qpos0
        scratch.qpos[self.qpos_adr] = self.q_home
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_camlight(model, scratch)
        R_ref = scratch.cam_xmat[self.cam_id].reshape(3, 3).copy()
        self.R_mount = scratch.xmat[self.head_body].reshape(3, 3).copy().T @ R_ref

        self._jacr = np.zeros((3, model.nv))
        self._scratch = mujoco.MjData(model)
        self.saturated = 0                      # diagnostics: steps spent against a joint limit
        self.steps = 0
        self._init_state(model, data)

    # ---- what a real IMU would hand us -------------------------------------------------
    def _imu(self, model, data):
        """Gravity direction and angular rate, both in the trunk frame. The robot reports both."""
        R_trunk = data.xmat[self.trunk_body].reshape(3, 3)
        g_trunk = -R_trunk.T[:, 2]              # == body_server.gravity_in_trunk(trunk quat)
        omega = data.qvel[self.trunk_dof + 3: self.trunk_dof + 6].copy()
        return g_trunk, omega

    def _aim_in_gat(self, model, data, W):
        """Where the head is pointing, expressed in the gravity-aligned trunk frame."""
        R_trunk = data.xmat[self.trunk_body].reshape(3, 3)
        R_unstab_world = data.xmat[self.head_body].reshape(3, 3) @ self.R_mount
        return W.T @ (R_trunk.T @ R_unstab_world)

    def _init_state(self, model, data):
        g, _ = self._imu(model, data)
        W = gravity_aligned_frame(g)
        A = self._aim_in_gat(model, data, W)
        # Start on the current aim so there is no startup transient.
        self.yaw_des = heading_of(A)
        self.R_tilt = _rz(-self.yaw_des) @ A

    def step(self, model, data, dt: float) -> None:
        g_trunk, omega = self._imu(model, data)
        W = gravity_aligned_frame(g_trunk)
        A = self._aim_in_gat(model, data, W)

        # Split the head's aim into heading-about-gravity and tilt. In this frame the split needs
        # no Euler convention: yaw is the rotation about z, tilt is whatever is left.
        yaw_head = heading_of(A, self.yaw_des)
        R_tilt_head = _rz(-yaw_head) @ A

        # TILT: track the head's mean aim slowly. Gait (1-2 Hz) is far above this corner, so it is
        # rejected outright; a deliberate change of aim still gets through within a few steps.
        k_aim = min(dt / max(self.tau_aim, 1e-6), 1.0)
        self.R_tilt = self.R_tilt @ _exp_so3(k_aim * _log_so3(self.R_tilt.T @ R_tilt_head))

        # YAW: hold inertially, then wash out. The gravity-aligned frame rotates WITH the trunk, so
        # holding a heading in space means counter-rotating here at the trunk's yaw rate -- the
        # component of the gyro about the gravity axis, which is all the sensor can give us. The
        # washout then bleeds toward wherever the robot has actually turned, so a sustained turn
        # passes through instead of saturating the joint.
        up = W[:, 2]
        yaw_rate = float(np.dot(omega, up))
        k_yaw = min(dt / max(self.tau_yaw, 1e-6), 1.0)
        self.yaw_des = _wrap(self.yaw_des - yaw_rate * dt)
        self.yaw_des = _wrap(self.yaw_des + k_yaw * _wrap(yaw_head - self.yaw_des))

        R_des_trunk = W @ (_rz(self.yaw_des) @ self.R_tilt)
        # Back to world for MuJoCo's world-frame Jacobian. This rotation appears on both sides of
        # the IK residual and cancels out of the solution, so no unobservable world heading enters
        # the joint angles -- the same q comes out of solving this entirely in the trunk frame.
        R_trunk = data.xmat[self.trunk_body].reshape(3, 3)
        R_des_full = R_trunk @ R_des_trunk

        R_cam = data.cam_xmat[self.cam_id].reshape(3, 3)
        R_des = R_cam @ _exp_so3(self.alpha * _log_so3(R_cam.T @ R_des_full))

        self._scratch.qpos[:] = data.qpos
        q = self.q_home.copy()
        for _ in range(self.iters):
            self._scratch.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(model, self._scratch)
            mujoco.mj_comPos(model, self._scratch)     # mj_jacBody reads cdof; kinematics alone
            mujoco.mj_camlight(model, self._scratch)   # does not fill it, and a zero Jacobian
            R_try = self._scratch.cam_xmat[self.cam_id].reshape(3, 3)   # looks exactly like success
            e = _log_so3(R_des @ R_try.T)
            if np.linalg.norm(e) < 1e-4:
                break
            mujoco.mj_jacBody(model, self._scratch, None, self._jacr, self.cam_body)
            J = self._jacr[:, self.dof_adr]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(3), e),
                        self.lo, self.hi)

        # Lead the solution by one actuator lag. q is a smooth function of a deterministic IK, so
        # a plain difference is well behaved; it is clipped back into range so the lead can never
        # command past a joint stop.
        q_cmd = q
        if self.tau_ff > 0.0 and self._q_prev is not None and dt > 0.0:
            q_cmd = np.clip(q + self.tau_ff * (q - self._q_prev) / dt, self.lo, self.hi)
        self._q_prev = q.copy()

        self.steps += 1
        if np.any(q <= self.lo + 1e-4) or np.any(q >= self.hi - 1e-4):
            self.saturated += 1
        data.ctrl[self.act] = q_cmd
