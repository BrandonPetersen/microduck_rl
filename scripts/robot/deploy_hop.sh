#!/bin/sh
# Put the HopFree hop policy on the robot. Runs on the LAPTOP.
#
#   sh scripts/robot/deploy_hop.sh [user@host]
#
# Backs up /etc/robot/robotd.toml first (the patcher stamps its own copy too),
# so `scripts/robot/restore_walk.sh` can put the walking setup back.
set -e
ROBOT="${1:-microduck@192.168.1.25}"
HERE=$(cd "$(dirname "$0")/../.." && pwd)
ONNX="$HERE/exports/hopfree_s50_dr_pf4eqwkv.onnx"

[ -f "$ONNX" ] || { echo "missing $ONNX"; exit 1; }
echo "target: $ROBOT"
ssh -o ConnectTimeout=8 "$ROBOT" true || { echo "robot unreachable"; exit 1; }

echo "== copying policy and driver"
scp "$ONNX" "$ROBOT:/tmp/"
scp "$HERE/scripts/hop_phase_driver.py" "$HERE/scripts/robot/start_hop_driver.sh" \
    "$HERE/scripts/robot/patch_robotd_toml.py" "$ROBOT:~/"
ssh "$ROBOT" 'chmod +x ~/start_hop_driver.sh; sudo cp /tmp/hopfree_s50_dr_pf4eqwkv.onnx /etc/robot/'

echo "== patching /etc/robot/robotd.toml"
# legs_lowpass is THE one to get right: the gait runs at ~8.5 Hz and a 0.85
# filter at 50 Hz cuts off near 1.4 Hz, attenuating it about sixfold. Training
# is unfiltered, so 0.0 is the matched value.
ssh "$ROBOT" 'sudo python3 ~/patch_robotd_toml.py \
    walk=\"hopfree_s50_dr_pf4eqwkv.onnx\" \
    stand=\"pose_home.onnx\" \
    gain=400 \
    action_scale=1.0 \
    legs_lowpass=0.0 \
    head_lowpass=0.5 \
    limp_fall=false \
    sitstand=\"none\" \
    cmd_alpha=1.0'

echo "== restarting robotd"
ssh "$ROBOT" 'sudo systemctl restart robotd && sleep 2 && systemctl is-active robotd'

cat <<'NEXT'

Deployed. On the robot:

    sh ~/start_hop_driver.sh 0.3      # 0.3 s hop bursts for the first test

START enables, A hops one burst, B relaxes.
Landing impacts are 3-6x body weight at ~8.5 Hz -- hold it or be ready to
catch, keep bursts short, and watch motor temperature.

Roll back with:  sh scripts/robot/restore_walk.sh
NEXT
