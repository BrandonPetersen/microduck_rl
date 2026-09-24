#!/bin/sh
# Launch the hop pad driver. Runs ON the robot. Installed to ~/start_hop_driver.sh
# by deploy_hop.sh -- NOT /tmp, which a power cycle wipes (it did, 2026-09-08).
#
# Two things this handles that a bare driver invocation does not:
#   * padd sends robot.move zeros every tick in Drive mode, which strobes the
#     applied twist against ours (measured 52/48 over 100 ticks) and looks like
#     "crazy motion". It has to be stopped for the duration.
#   * the pad's /dev/input/eventN renumbers across reboots, so it is found by
#     name rather than assumed.
#
# Buttons: START enables, A hops one burst, B relaxes.
set -e
BURST="${1:-1.0}"     # seconds of hopping per A press; 0.3 is a cautious first test

echo "stopping padd (it fights us for robot.move)"
sudo systemctl stop padd || true

DEV=""
for e in /dev/input/event*; do
  NAME=$(cat "/sys/class/input/$(basename "$e")/device/name" 2>/dev/null || echo "")
  case "$NAME" in
    *Pro\ Controller*|*Gamepad*|*Xbox*|*8BitDo*|*Controller*)
      DEV="$e"; echo "pad: $e  ($NAME)"; break ;;
  esac
done
[ -n "$DEV" ] || { echo "no pad found; is it paired and on?"; exit 1; }

echo "hop burst: ${BURST}s per A press"
echo "START = enable   A = hop   B = relax   ctrl-C = quit"
exec python3 ~/hop_phase_driver.py --device "$DEV" --hold 0.65 \
     --hop-seconds "$BURST" --enable-bit
