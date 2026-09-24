#!/bin/sh
# Put the HopFree hop policy on the robot. Runs on the LAPTOP.
#
#   sh scripts/robot/deploy_hop.sh [user@host]
#
# Roll back with scripts/robot/restore_walk.sh.
set -e
ROBOT="${1:-microduck@192.168.1.25}"
HERE=$(cd "$(dirname "$0")/../.." && pwd)
ONNX="$HERE/exports/hopfree_s50_dr_pf4eqwkv.onnx"
RUNTIME="$HERE/../microduck_runtime/policies"

[ -f "$ONNX" ] || { echo "missing $ONNX"; exit 1; }
ssh -o ConnectTimeout=8 "$ROBOT" true || { echo "robot unreachable"; exit 1; }
echo "== target: $ROBOT"

ssh "$ROBOT" 'rm -f /tmp/*.onnx'
scp "$ONNX" "$ROBOT:/tmp/"
# Ship the walking pair too: neither is on this robot, so without them the
# rollback would point robotd at a file that does not exist. Both are symlinks
# locally; scp copies the contents.
for w in alpha_walking.onnx alpha_stand.onnx; do
  [ -f "$RUNTIME/$w" ] && scp "$RUNTIME/$w" "$ROBOT:/tmp/$w"
done
scp "$HERE/scripts/hop_phase_driver.py" \
    "$HERE/scripts/robot/start_hop_driver.sh" \
    "$HERE/scripts/robot/robot_side_deploy.sh" \
    "$HERE/scripts/robot/patch_robotd_toml.py" "$ROBOT:~/"

ssh "$ROBOT" 'chmod +x ~/start_hop_driver.sh ~/robot_side_deploy.sh; ~/robot_side_deploy.sh hop'

cat <<'NEXT'

Deployed. On the robot:

    sh ~/start_hop_driver.sh 0.3      # 0.3 s hop bursts for the first test

START enables, A hops one burst, B relaxes.
Landing impacts are 3-6x body weight at ~8.5 Hz -- hold it or be ready to
catch, keep bursts short, and watch motor temperature.

Roll back with:  sh scripts/robot/restore_walk.sh
NEXT
