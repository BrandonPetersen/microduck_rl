"""Put the camera on its own 2-axis gimbal, instead of stabilising it with the neck.

WHY. duckslam's ladder found that vision-only odometry fails on this robot because the gait rotates
the camera far more than it translates it (26 px of rotational optical flow against 5 px of
parallax), and that removing that rotation takes vision-only ATE from 249 cm to 15 cm. The obvious
way to remove it -- drive the neck as a gimbal -- was tried and does not work, for a reason that
has nothing to do with optics:

    the walking policy NEEDS its head. It moves the neck 8.9 deg / 6.1 deg while walking and is
    rewarded for tracking a head-pose command. Commandeering those four joints pinned the head at
    2.3 deg / 1.0 deg, and the duck walked 9.76 m and then stalled, standing, while being commanded
    0.12 m/s for another 180 s. A gimbal-off control on the same day, same procedure, walked 30.57 m.

So the camera needs its own degrees of freedom. This inserts a small gimbal body between the head
shell and the camera: the head keeps doing whatever the policy wants, and the camera is held level
independently. Three things make this strictly better than using the neck:

  * the policy is untouched -- these joints are named outside `JOINT_NAMES`, so the wire protocol
    has no slot for them and `robotd` can neither see nor command them;
  * the load collapses. The neck swings the entire head assembly; a camera gimbal moves a camera.
    The kp=0.55 Nm/rad authority that made the neck gimbal achieve only 1.04x rejection is not a
    constraint when the thing being rotated weighs a few grams;
  * it is ordinary hardware. A 2-axis camera gimbal is a cheap, common part -- a far smaller ask
    than uprating the neck servos of a walking biped.

BUILT AT MODEL-COMPILE TIME, not in the XML, because the robot MJCF is shared with training and
every other task, and none of them should grow two joints because a SLAM experiment wanted them.

THE EXTRINSIC MOVES. The camera is no longer rigid to the head, so anything reconstructing the
camera pose from the robot's kinematics must read these joint angles too. In simulation the bench
already records the true camera pose, so nothing downstream changes; on hardware this is what the
gimbal's own encoders are for. Stated here because it is the one thing this design costs.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Deliberately NOT in body_server.JOINT_NAMES: that tuple is the wire protocol, indexed positionally
# by every array robotd sends and receives. A joint absent from it cannot be commanded by the
# daemon, which is exactly the property this design needs.
# THREE axes, not two. A 2-axis gimbal was measured at only 2.1x rejection and -- decisively --
# raising its actuator gain 25x did not improve it (2.14 -> 2.18x). That is not a torque limit, it
# is a rank limit: two joints cannot correct three rotation components, and the ~3.1 deg that
# survives is simply the direction they do not span. Three axes is also what a real camera gimbal
# has, so this costs nothing in realism.
GIMBAL_JOINTS = ("cam_pitch", "cam_roll", "cam_yaw")
GIMBAL_AXES = {"cam_pitch": (0.0, 1.0, 0.0), "cam_roll": (1.0, 0.0, 0.0),
               "cam_yaw": (0.0, 0.0, 1.0)}
CAMERA = "head_camera"
# Enough travel for the measured demand (~+-5 deg, docs/validation/data/08-neck-demand.txt in
# microduck_vslam) with room to spare, and little enough that a runaway cannot point the camera
# somewhere absurd.
RANGE_RAD = 0.52                      # +-30 deg
KP = 30.0                             # a camera is grams; this is a small servo, not a neck one
FORCE = 2.0                           # Nm, generous for the load


def add_camera_gimbal(spec: mujoco.MjSpec, prefix: str = "") -> None:
    """Re-parent `head_camera` onto a 2-DOF gimbal body, in place, on an uncompiled spec.

    Three hinges about the camera's own centre, so the mount can express any small rotation. Yaw is
    included despite heading being the robot's to choose, because the controller only ever asks for
    the heading the trunk already has -- it is there to complete the basis, not to steer.
    """
    cam = None
    for body in spec.bodies:
        for c in body.cameras:
            if c.name == prefix + CAMERA:
                cam, host = c, body
                break
        if cam is not None:
            break
    if cam is None:
        raise RuntimeError(f"no camera named {prefix + CAMERA!r} in the spec")

    pos = np.array(cam.pos, dtype=float).copy()
    quat = np.array(cam.quat, dtype=float).copy()
    name, fovy, resolution = cam.name, cam.fovy, np.array(cam.resolution).copy()
    # The original is RENAMED rather than removed, and that is deliberate twice over. A spec will
    # not hold two cameras of the same name, so the canonical name has to be freed before the
    # gimballed one can take it -- and what is left behind is useful: a camera still rigidly bolted
    # to the head, at the same place, so one capture can render the stabilised and unstabilised
    # views of the very same walk. That is a paired comparison no two runs can give, because the
    # gait is stochastic.
    cam.name = f"{prefix}{CAMERA}_rigid"

    # The gimbal body sits exactly where the camera did, so the optical centre does not move and
    # the whole intervention stays a pure rotation about it -- which is what makes it equivalent to
    # the warp the offline experiments measured.
    g = host.add_body(name=f"{prefix}cam_gimbal", pos=pos, quat=[1.0, 0.0, 0.0, 0.0])
    for j in GIMBAL_JOINTS:
        g.add_joint(name=f"{prefix}{j}", type=mujoco.mjtJoint.mjJNT_HINGE,
                    axis=list(GIMBAL_AXES[j]), range=[-RANGE_RAD, RANGE_RAD])
    # A token inertial: MuJoCo needs mass on a body carrying joints, and the real part is light.
    g.add_geom(name=f"{prefix}cam_gimbal_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[0.004, 0.004, 0.004], mass=0.005, contype=0, conaffinity=0, group=3)
    # The camera moves onto the gimbal at zero offset, keeping the orientation it had.
    g.add_camera(name=name, pos=[0.0, 0.0, 0.0], quat=quat, fovy=fovy, resolution=resolution)

    for j in GIMBAL_JOINTS:
        a = spec.add_actuator(name=f"{prefix}{j}", target=f"{prefix}{j}",
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = KP
        a.biasprm[1] = -KP
        a.forcerange = [-FORCE, FORCE]
        a.ctrlrange = [-RANGE_RAD, RANGE_RAD]


class CameraGimbalStabilizer:
    """Hold the camera level using its own two joints, leaving the neck to the walking policy.

    Same absolute-IK formulation as `head_stabilizer.HeadStabilizer` and for the same reasons --
    solving for joint angles from home every step has no accumulated state, so it can neither drift
    up the null space nor ring against a lagging actuator, both of which the neck version did before
    it was fixed. The differences are only that there are two joints instead of four, and that no
    engage gate is needed: these joints are not shared with any other controller, so a fallen or
    sitting duck loses nothing by having its camera held level.
    """

    def __init__(self, model, data, *, prefix: str = "", alpha: float = 1.0,
                 damping: float = 1e-4, iters: int = 4, yaw_tau: float = 0.5):
        from .head_stabilizer import _rz, _yaw_of
        self.alpha, self.damping, self.iters = float(alpha), float(damping), int(iters)
        self.yaw_tau = float(yaw_tau)
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, prefix + CAMERA)
        if self.cam_id < 0:
            raise RuntimeError(f"no camera {prefix + CAMERA!r}; was add_camera_gimbal() applied?")
        self.cam_body = int(model.cam_bodyid[self.cam_id])
        self.trunk_body = int(model.body_rootid[self.cam_body])
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)
                for j in GIMBAL_JOINTS]
        self.act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + j)
                             for j in GIMBAL_JOINTS])
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0].copy()
        self.hi = model.jnt_range[jids, 1].copy()
        self.q_home = np.zeros(len(jids))

        # The level reference, from geometry rather than from whatever pose the duck starts in --
        # reading it live is what pinned the neck gimbal to the SIT pose and stopped the robot
        # standing for a whole 250 s capture.
        mujoco.mj_forward(model, data)
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = model.qpos0
        scratch.qpos[self.qpos_adr] = self.q_home
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_camlight(model, scratch)
        R_ref = scratch.cam_xmat[self.cam_id].reshape(3, 3).copy()
        R_trunk_ref = scratch.xmat[self.trunk_body].reshape(3, 3).copy()
        self.R0 = _rz(-_yaw_of(R_trunk_ref)) @ R_ref
        self.yaw_f = _yaw_of(data.xmat[self.trunk_body].reshape(3, 3))
        self._jacr = np.zeros((3, model.nv))
        self._scratch = mujoco.MjData(model)

    def step(self, model, data, dt: float) -> None:
        from .head_stabilizer import _exp_so3, _log_so3, _rz, _yaw_of
        R_cam = data.cam_xmat[self.cam_id].reshape(3, 3)
        R_trunk = data.xmat[self.trunk_body].reshape(3, 3)
        # **The heading must be FILTERED, not followed.** Tracking the trunk's yaw directly was
        # measured putting 1.327 deg/frame of gait wobble into the setpoint -- the gimbal then
        # chased a shaking goal and rejected only 1.11x of the camera's rotation despite swinging
        # its joints 11-17 deg. The robot's heading genuinely turns (356.8 deg over one lap), so
        # the filter has to follow the slow turn while rejecting the per-step wobble, which is
        # what a time constant of a few gait cycles does.
        yaw = _yaw_of(R_trunk)
        k = min(dt / max(self.yaw_tau, 1e-6), 1.0)
        self.yaw_f += np.arctan2(np.sin(yaw - self.yaw_f), np.cos(yaw - self.yaw_f)) * k
        R_des_full = _rz(self.yaw_f) @ self.R0
        R_des = R_cam @ _exp_so3(self.alpha * _log_so3(R_cam.T @ R_des_full))

        self._scratch.qpos[:] = data.qpos
        q = self.q_home.copy()
        for _ in range(self.iters):
            self._scratch.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(model, self._scratch)
            mujoco.mj_comPos(model, self._scratch)   # mj_jacBody reads cdof; kinematics alone
            mujoco.mj_camlight(model, self._scratch)  # does not fill it, and a zero Jacobian
            R_try = self._scratch.cam_xmat[self.cam_id].reshape(3, 3)  # looks exactly like success
            e = _log_so3(R_des @ R_try.T)
            if np.linalg.norm(e) < 1e-4:
                break
            mujoco.mj_jacBody(model, self._scratch, None, self._jacr, self.cam_body)
            J = self._jacr[:, self.dof_adr]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(3), e),
                        self.lo, self.hi)
        data.ctrl[self.act] = q
