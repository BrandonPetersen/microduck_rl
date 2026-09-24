#!/bin/sh
# Measure joint travel with the bus actually free. Runs ON the robot.
#
#   sudo sh ~/measure_travel.sh
#
# WHY A WRAPPER. `systemctl stop robotd` does not keep robotd stopped: the unit
# is Restart=always with RestartSec=2s, and updaterd's health gate restarts it
# besides (`on_apply = { action = "restart", units = ["robotd"] }`). Measured:
# stopped at t+0, inactive t+1..t+3, activating t+4, active t+5. So a sweep
# longer than about three seconds has robotd back on the bus contending for
# every reply -- which is exactly what produced one good cycle followed by a
# stream of desynced reads, three times over.
#
# Masking blocks the restart. The trap puts everything back on any exit path,
# including the Ctrl-C that ends the sweep.
#
# NO `set -e` HERE, deliberately. `systemctl mask` returns non-zero even when it
# succeeds, so set -e fired the cleanup trap before the sweep ever ran -- twice.
# A wrapper whose whole job is "restore state on every exit path" wants an
# explicit trap, not an implicit abort on every non-zero status.
restore() {
  echo
  echo "restoring robotd..."
  sudo systemctl unmask robotd >/dev/null 2>&1 || true
  sudo systemctl start robotd >/dev/null 2>&1 || true
  sleep 2
  echo "robotd: $(systemctl is-active robotd)"
}
trap restore EXIT INT TERM

sudo systemctl mask robotd >/dev/null 2>&1 || true
sudo systemctl stop robotd >/dev/null 2>&1 || true
sleep 1
# `[ ... ] && { ... }` here, under set -e, exits the script when the test is
# FALSE -- i.e. on the success path -- and the sweep never ran.
if [ "$(systemctl is-active robotd)" = "active" ]; then
  echo "robotd would not stop"
  exit 1
fi
echo "bus is free (robotd masked for the duration)"
echo
# Not `~/joint_travel.py`: under sudo that expands to /root. Resolve it next to
# this script instead, so it works however the wrapper is invoked.
DIR=$(cd "$(dirname "$0")" && pwd)
sudo python3 "$DIR/joint_travel.py" "$@"
