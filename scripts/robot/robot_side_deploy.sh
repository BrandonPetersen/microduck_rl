#!/bin/sh
# Runs ON the robot, invoked by deploy_hop.sh. Kept as its own file rather than
# an inline ssh string because the nested quoting of a multi-key config patch
# through ssh is a reliable way to write the wrong value.
#
#   robot_side_deploy.sh hop     -- install the HopFree hop policy
#   robot_side_deploy.sh walk    -- restore the walking setup
set -e
POLDIR=/opt/robot/policies/current   # this robot names policies by ABSOLUTE path
MODE="${1:-hop}"

for f in /tmp/*.onnx; do
  [ -f "$f" ] && sudo cp "$f" "$POLDIR/" && echo "installed $(basename "$f")"
done

if [ "$MODE" = "hop" ]; then
  # legs_lowpass is the one that matters: the gait runs at ~8.5 Hz and 0.85 at
  # 50 Hz cuts off near 1.4 Hz, attenuating it about sixfold. Training is
  # unfiltered, so 0.0 is the matched value.
  sudo python3 ~/patch_robotd_toml.py \
      walk="\"$POLDIR/hopfree_s50_dr_pf4eqwkv.onnx\"" \
      stand="\"$POLDIR/pose_home.onnx\"" \
      gain=400 action_scale=1.0 \
      legs_lowpass=0.0 head_lowpass=0.5 \
      limp_fall=false cmd_alpha=1.0 \
      sitstand='"none"'
else
  sudo python3 ~/patch_robotd_toml.py \
      walk="\"$POLDIR/alpha_walking.onnx\"" \
      stand="\"$POLDIR/alpha_stand.onnx\"" \
      gain=200 action_scale=0.9 \
      legs_lowpass=0.7 head_lowpass=0.5 \
      limp_fall=true cmd_alpha=1.0
  sudo systemctl start padd || true
fi

sudo systemctl restart robotd
sleep 2
echo "robotd: $(systemctl is-active robotd)   padd: $(systemctl is-active padd)"
grep -E "^(walk|stand|gain|action_scale|legs_lowpass|limp_fall|sitstand) " /etc/robot/robotd.toml
