#!/bin/sh
# Put the walking setup back after a hop test. Runs on the LAPTOP.
#
#   sh scripts/robot/restore_walk.sh [user@host]
#
# Sets the walking values EXPLICITLY rather than restoring the newest backup:
# on this robot the newest backup has been a previous *hop* config since
# 2026-09-08, so "restore the backup" would not restore walking. Values are the
# daemon's own documented defaults (microduck_daemon/deploy/robotd.toml).
set -e
ROBOT="${1:-microduck@192.168.1.25}"
HERE=$(cd "$(dirname "$0")/../.." && pwd)
ssh -o ConnectTimeout=8 "$ROBOT" true || { echo "robot unreachable"; exit 1; }
scp "$HERE/scripts/robot/robot_side_deploy.sh" \
    "$HERE/scripts/robot/patch_robotd_toml.py" "$ROBOT:~/" >/dev/null
ssh "$ROBOT" 'chmod +x ~/robot_side_deploy.sh; ~/robot_side_deploy.sh walk'
echo
echo "walking setup restored, pad handed back to padd."
