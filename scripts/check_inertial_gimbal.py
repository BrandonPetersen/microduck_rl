"""Which frequencies does each gimbal controller actually reject?

The whole claim for `InertialGimbalStabilizer` is about WHERE in frequency the two controllers
differ, so a single "is it better" number would hide the answer. This imposes a known sinusoidal
trunk tilt at several frequencies and measures how much of it survives at the camera.

    rejection = trunk tilt amplitude / camera tilt amplitude        (higher is better)

Expected shape, and what would falsify the design:

  * gait band (1-2 Hz): both should reject. The follow controller already did -- that is why its
    video looks stable.
  * slow band (< 0.2 Hz): the follow controller should reject almost NOTHING, because its reference
    is a 0.5 s low-pass of the head and anything slower passes through by construction. The
    inertial one should still reject, because gravity says the horizon did not move. If it does
    not, the design is wrong and the rest of the work is pointless.

The duck is held at its home pose with gravity off and the trunk driven kinematically, so this
measures the CONTROLLER and not the walking policy. Sensed gravity is computed from the trunk's
orientation, not from `opt.gravity`, so switching physics gravity off does not blind the controller.

    uv run python scripts/check_inertial_gimbal.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def axis_angle(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(c))


def run(model, data, ctrl, freq, amp, dt, warmup, measure, trunk_qadr, trunk_dadr, cam_id):
    """Drive the trunk in pitch at `freq`, return (trunk amplitude, camera amplitude) in radians.

    `warmup` must exceed several aim time constants. The first version ran 2 Hz for 4 s total and
    measured the back half -- but the aim filter has tau = 5 s, so it was still settling, and the
    controller was charged for a startup transient. That alone made it look WORSE than the
    follow controller in the gait band.
    """
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    stab = ctrl(model, data)

    R0_cam = None
    tr, cam = [], []
    n = int((warmup + measure) / dt)
    for i in range(n):
        t = i * dt
        ang = amp * np.sin(2.0 * np.pi * freq * t)
        rate = amp * 2.0 * np.pi * freq * np.cos(2.0 * np.pi * freq * t)
        # Impose trunk pitch kinematically, and hand the gyro the matching rate.
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0.0, 1.0, 0.0]), ang)
        data.qpos[trunk_qadr + 3: trunk_qadr + 7] = q
        data.qvel[trunk_dadr + 3: trunk_dadr + 6] = [0.0, rate, 0.0]
        mujoco.mj_forward(model, data)

        stab.step(model, data, dt)
        mujoco.mj_step(model, data)

        if t >= warmup:                             # discard the settling transient
            R_cam = data.cam_xmat[cam_id].reshape(3, 3).copy()
            if R0_cam is None:
                R0_cam = R_cam
            tr.append(abs(ang))
            cam.append(axis_angle(R0_cam.T @ R_cam))
    return float(np.max(tr)), float(np.max(cam)), stab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="src/mjlab_microduck/robot/microduck/scene_vslam.xml")
    ap.add_argument("--amp-deg", type=float, default=6.0, help="imposed trunk tilt amplitude")
    ap.add_argument("--freqs", default="0.05,0.1,0.25,0.5,1.0,2.0")
    ap.add_argument("--cycles", type=float, default=3.0, help="cycles measured at each frequency")
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="settling seconds before measuring (default: 5 aim time constants)")
    args = ap.parse_args()

    from mjlab_microduck.sim.body_server import build_world
    from mjlab_microduck.sim.camera_gimbal import CAMERA, CameraGimbalStabilizer
    from mjlab_microduck.sim.inertial_gimbal import InertialGimbalStabilizer

    model = build_world(Path(args.scene), 1, camera_gimbal=True)
    model.opt.gravity[:] = 0.0                       # keep the duck still; sensing is from the quat
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    trunk_body = int(model.body_rootid[int(model.cam_bodyid[cam_id])])
    trunk_qadr = int(model.body_jntadr[trunk_body])
    trunk_qadr = int(model.jnt_qposadr[trunk_qadr])
    trunk_dadr = int(model.body_dofadr[trunk_body])

    amp = np.radians(args.amp_deg)
    freqs = [float(x) for x in args.freqs.split(",")]
    makers = {"follow (current)": lambda m, d: CameraGimbalStabilizer(m, d),
              "inertial (new)": lambda m, d: InertialGimbalStabilizer(m, d)}

    print(f"scene={Path(args.scene).name}  trunk tilt +-{args.amp_deg}deg  dt={dt*1000:.1f} ms\n")
    print(f"{'freq':>7} {'':>4} " + " ".join(f"{k:>22}" for k in makers))
    print(f"{'Hz':>7} {'':>4} " + " ".join(f"{'residual   rejection':>22}" for _ in makers))

    rows = {}
    from mjlab_microduck.sim.inertial_gimbal import TAU_AIM
    warmup = args.warmup or 5.0 * TAU_AIM
    print(f"warmup {warmup:.0f} s per point (>= 5 aim time constants), "
          f"then {args.cycles:.0f} cycles measured\n")
    for f in freqs:
        measure = max(args.cycles / f, 3.0)
        cells, res = [], {}
        for name, mk in makers.items():
            a_tr, a_cam, stab = run(model, data, mk, f, amp, dt, warmup, measure,
                                    trunk_qadr, trunk_dadr, cam_id)
            rej = a_tr / max(a_cam, 1e-9)
            res[name] = rej
            cells.append(f"{np.degrees(a_cam):8.2f}d {rej:9.1f}x")
        rows[f] = res
        print(f"{f:7.2f} {'':>4} " + " ".join(cells))

    print("\nrejection = imposed trunk tilt / surviving camera tilt; higher is better.")
    slow = [f for f in freqs if f <= 0.25]
    if slow:
        a = np.mean([rows[f]["follow (current)"] for f in slow])
        b = np.mean([rows[f]["inertial (new)"] for f in slow])
        print(f"\nslow band (<=0.25 Hz), the band that accumulates into trajectory error:")
        print(f"  follow   {a:6.1f}x")
        print(f"  inertial {b:6.1f}x   -> {b/max(a,1e-9):.1f}x better")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
