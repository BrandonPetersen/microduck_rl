"""Is the gimbal's residual rotation SMOOTH or jittery? Tune offline before spending a capture.

The measure that matters is not how much rotation is left but whether what is left is predictable:
lag-1 autocorrelation of the inter-frame rotation vector. The software warp leaves +0.548 (smooth);
the gimbal as first built left +0.065 (white noise), and performed 4-6x worse in ATE despite a
SMALLER residual.
"""
import sys, numpy as np, mujoco
sys.path.insert(0,'/home/steve/Project/Repo/Pollen/MICRODUCK/microduck_rl/src')
from pathlib import Path
from mjlab_microduck.sim.body_server import World, Body, pose_table, TIMESTEP
import mjlab_microduck.sim.camera_gimbal as CG
SC=Path('/home/steve/Project/Repo/Pollen/MICRODUCK/microduck_rl/src/mjlab_microduck/robot/microduck/scene_vslam.xml')
FPS_EVERY=int(round(1/(16.67*TIMESTEP)))   # sample at the capture's frame rate

def log_so3(R):
    c=np.clip((np.trace(R)-1)/2,-1,1); th=float(np.arccos(c))
    if th<1e-9: return np.zeros(3)
    w=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]]); return w/(2*np.sin(th))*th

def run(gimbal_on, seconds=14.0):
    w=World(SC,1,camera_gimbal=True)
    pose,tz=pose_table(SC,'STAND'); b=Body(w,0,limp=False); b.place(pose,tz,offset_y=0.0); w.bodies.append(b)
    mujoco.mj_forward(w.model,w.data)
    gs=CG.CameraGimbalStabilizer(w.model,w.data,prefix=b.prefix)
    if gimbal_on: b.head_stab=gs
    b.released=True
    cid=gs.cam_id
    tr=int(w.model.jnt_qposadr[mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_JOINT,'trunk_base_freejoint')])
    td=int(w.model.jnt_dofadr[mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_JOINT,'trunk_base_freejoint')])
    q=np.zeros(4); Rs=[]
    for i in range(int(seconds/TIMESTEP)):
        t=i*TIMESTEP; a=np.radians(4.0)
        # gait-like trunk oscillation, plus a slow turn so the aim genuinely moves
        mujoco.mju_euler2Quat(q, np.array([a*np.sin(2*np.pi*1.4*t), a*np.sin(2*np.pi*1.4*t+1.1), 0.15*t]),'xyz')
        w.data.qpos[tr+3:tr+7]=q; w.data.qvel[td:td+6]=0.0
        w.step(1)
        if i % FPS_EVERY == 0: Rs.append(w.data.cam_xmat[cid].reshape(3,3).copy())
    Rs=np.array(Rs[len(Rs)//3:])          # drop the settling transient
    wv=np.array([log_so3(Rs[i].T@Rs[i+1]) for i in range(len(Rs)-1)])
    mag=np.degrees(np.linalg.norm(wv,axis=1))
    a_,b_=wv[:-1].ravel(), wv[1:].ravel()
    return np.median(mag), float(np.corrcoef(a_,b_)[0,1])

print(f"{'config':18} {'median rot/frame':>17} {'lag-1 autocorr':>16}")
for on,lbl in ((False,'gimbal off'),(True,'gimbal on')):
    m,ac=run(on)
    print(f'{lbl:18} {m:16.3f}d {ac:+15.3f}')
print('\ntarget: autocorrelation near the warp\'s +0.55, i.e. a SMOOTH residual')
