#!/bin/sh
# Put the walking setup back after a hop test. Runs on the LAPTOP.
#
#   sh scripts/robot/restore_walk.sh [user@host]
#
# Sets the walking configuration EXPLICITLY rather than restoring the newest
# backup, because the newest backup is whatever the robot happened to be in
# before the hop deploy -- and on this robot that has been a *previous hop*
# config since 2026-09-08, not a walking one. Values are the daemon's own
# documented defaults from microduck_daemon/deploy/robotd.toml.
#
# If this robot needs different per-robot values, the timestamped backups are
# in /etc/robot/robotd.toml.bak-* and this script prints them.
set -e
ROBOT="${1:-microduck@192.168.1.25}"
ssh -o ConnectTimeout=8 "$ROBOT" true || { echo "robot unreachable"; exit 1; }

ssh "$ROBOT" 'sudo python3 ~/patch_robotd_toml.py \
    walk=\"alpha_walking.onnx\" \
    stand=\"alpha_stand.onnx\" \
    sitstand=\"alpha_sitstand.onnx\" \
    gain=200 \
    action_scale=0.9 \
    legs_lowpass=0.7 \
    head_lowpass=0.5 \
    limp_fall=true \
    cmd_alpha=1.0'

ssh "$ROBOT" 'sudo systemctl restart robotd
sudo systemctl start padd || true
sleep 2
echo "robotd: $(systemctl is-active robotd)   padd: $(systemctl is-active padd)"
echo "backups available:"; ls -1t /etc/robot/robotd.toml.bak-* 2>/dev/null | head -5'
echo "walking setup restored, pad handed back to padd."
