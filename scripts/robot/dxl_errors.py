#!/usr/bin/env python3
"""Read (and optionally clear) Dynamixel Hardware Error Status. Runs ON the robot.

Nothing shipped on the robot can do this: `robotctl health` reports the bus as
"ok" and the motors as cool while the servos themselves are blinking red,
because a latched Hardware Error Status lives in the servo and is not surfaced.
A latched servo refuses torque until it is rebooted, which is what a robot that
"will not go to the standing position" looks like.

    sudo systemctl stop robotd            # the daemon owns the bus
    sudo python3 dxl_errors.py            # read every id
    sudo python3 dxl_errors.py --reboot   # clear the latch on the ones that erred
    sudo systemctl start robotd

Protocol 2.0, 1 Mbps (model.rs: baud_rate register 3 = 1 Mbps).
"""
import argparse
import struct
import sys
import time

ADDR_HW_ERROR, ADDR_VOLT, ADDR_TEMP = 70, 144, 146
# duck-control/src/model.rs JOINT_IDS, in JOINT_NAMES order:
# left leg (5), neck/head/mouth (5), right leg (5).
NAMES = {
    20: "left_hip_yaw", 21: "left_hip_roll", 22: "left_hip_pitch",
    23: "left_knee", 24: "left_ankle",
    30: "neck_pitch", 31: "head_pitch", 32: "head_yaw", 33: "head_roll", 34: "mouth",
    10: "right_hip_yaw", 11: "right_hip_roll", 12: "right_hip_pitch",
    13: "right_knee", 14: "right_ankle",
}
# The daemon pins shutdown = 52 on every servo (model.rs EXPECTED_REGISTERS):
# bits 2, 4 and 5. So overheating, electrical shock and OVERLOAD latch torque
# off and the joint refuses to move until rebooted -- but bit 0, input voltage,
# does NOT. That distinction is the whole diagnosis: every servo on this robot
# carries bit 0 permanently (2S pack at ~8 V), so blinking red LEDs are its
# normal state and never explain a robot that will not stand.
LATCHING = 52
# The joints whose MJCF range is exactly +/-pi/2 on both sides -- an
# onshape-to-robot export default rather than a measured limit. These are what
# `--watch` measures by default, because they are the ones the model invented.
# duck-control model.rs DEFAULT_POSITION, in degrees. bus.rs maps raw ticks
# straight to joint angle -- (2*pi*raw/4096) - pi, no per-joint sign or offset
# -- so a measured sweep is directly comparable to these. Every joint MUST be
# able to reach its home angle, which makes this the sanity check on a sweep.
HOME_DEG = {
    20: 0.0, 21: -5.0, 22: -26.2, 23: -0.3, 24: +26.0,
    30: +20.0, 31: +20.0, 32: 0.0, 33: 0.0, 34: 0.0,
    10: 0.0, 11: +5.0, 12: +26.2, 13: +0.3, 14: -26.0,
}
# Mirrored pairs: the legs are equal and opposite in this convention, so a
# left span [a, b] predicts a right span [-b, -a]. Disagreement means one of
# the two was not swept fully.
MIRROR = {20: 10, 21: 11, 22: 12, 23: 13, 24: 14,
          10: 20, 11: 21, 12: 22, 13: 23, 14: 24}
PLACEHOLDER_JOINTS = "left_ankle,right_ankle,left_knee,right_knee,left_hip_pitch,right_hip_pitch"
SHORT = {"left_ankle": "l_ank", "right_ankle": "r_ank", "left_knee": "l_kne",
         "right_knee": "r_kne", "left_hip_pitch": "l_hip", "right_hip_pitch": "r_hip"}
INST_READ, INST_REBOOT = 0x02, 0x08
# Hardware Error Status bits, XL330 control table.
BITS = {
    0: "input voltage out of range",
    2: "OVERHEATING",
    3: "motor encoder",
    4: "electrical shock / short",
    5: "OVERLOAD (sustained torque above limit)",
}


class _Port:
    """Raw 8N1 serial via termios. Stdlib only: the robot has no pyserial, and
    installing packages onto it to read one register is the wrong trade."""

    def __init__(self, path: str, baud: int):
        import fcntl
        import os
        import termios
        self.os, self.termios = os, termios
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        speed = getattr(termios, f"B{baud}")
        # iflag oflag cflag lflag ispeed ospeed cc
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[4] = attrs[5] = speed
        attrs[6] = list(attrs[6])
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        del fcntl

    @property
    def in_waiting(self) -> int:
        import array
        import fcntl
        buf = array.array("i", [0])
        fcntl.ioctl(self.fd, termios.FIONREAD if hasattr(self, "_x") else 0x541B, buf)
        return buf[0]

    def reset_input_buffer(self) -> None:
        self.termios.tcflush(self.fd, self.termios.TCIFLUSH)

    def write(self, data: bytes) -> None:
        # O_NONBLOCK means a full kernel tx buffer raises EAGAIN rather than
        # waiting. At 1 Mbps with six joints polled in a loop that happens
        # readily, and an unhandled EAGAIN killed a live measurement mid-run.
        import select
        view = memoryview(data)
        deadline = time.time() + 0.2
        while view:
            try:
                n = self.os.write(self.fd, view)
                view = view[n:]
            except BlockingIOError:
                if time.time() > deadline:
                    raise
                select.select([], [self.fd], [], 0.01)

    def flush(self) -> None:
        self.termios.tcdrain(self.fd)

    def read(self, n: int) -> bytes:
        try:
            return self.os.read(self.fd, max(n, 1))
        except BlockingIOError:
            return b""


def crc16(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def packet(dxl_id: int, inst: int, params: bytes = b"") -> bytes:
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, dxl_id]) + struct.pack("<H", len(params) + 3) + bytes([inst]) + params
    return body + struct.pack("<H", crc16(body))


def _status_frames(buf: bytes):
    """Yield (id, err, params) for every VALID status packet in buf.

    THE ECHO IS THE WHOLE PROBLEM. The bus is half duplex, so every byte we
    transmit comes straight back, header and all. A parser that locks onto the
    first 0xFF 0xFF 0xFD 0x00 it sees decodes our own READ instruction as if it
    were a reply and reports whatever follows as register data -- which is how
    an earlier version of this script produced 6552 V, 78 C on a cold robot,
    and three mutually contradictory scans of an unchanged machine.

    Three checks make a frame real: the instruction byte must be 0x55 (STATUS,
    not our 0x02 READ), the CRC must verify, and the id must be the one asked.
    """
    i = 0
    while True:
        i = buf.find(b"\xff\xff\xfd\x00", i)
        if i < 0 or len(buf) < i + 7:
            return
        length = struct.unpack("<H", buf[i + 5:i + 7])[0]
        end = i + 7 + length
        if length < 4 or len(buf) < end:
            i += 4
            continue
        frame = buf[i:end]
        if frame[7] == 0x55 and crc16(frame[:-2]) == struct.unpack("<H", frame[-2:])[0]:
            yield frame[4], frame[8], frame[9:-2]
        i += 4


def txrx(ser, dxl_id: int, pkt: bytes, expect: int, timeout: float = 0.15, tries: int = 3):
    """NO ECHO DRAIN. This board's transceiver handles direction, so our own
    packet does NOT come back -- draining len(pkt) bytes ate the reply instead
    and every register read came back as zero. Validated the other way: without
    the drain, temperatures and voltages match `robotctl health` exactly.
    _status_frames rejects anything that is not a CRC-valid status packet from
    the id we asked, which is what makes that safe."""
    for _ in range(tries):
        ser.reset_input_buffer()
        ser.write(pkt)
        ser.flush()
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            chunk = ser.read(64)
            if chunk:
                buf += chunk
                for fid, err, params in _status_frames(buf):
                    if fid == dxl_id and len(params) >= expect:
                        return err, params[:expect]
            else:
                time.sleep(0.001)
    return None, None


def _watch(ser, ids, names):
    import signal
    lo = {i: 9e9 for i in ids}
    hi = {i: -9e9 for i in ids}
    n_ok = {i: 0 for i in ids}      # a joint that never answers must SAY so
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
    print("Torque must be OFF (robotctl robot relax). Move each joint slowly to BOTH stops.")
    print("Ctrl-C when done.\n")
    while not stop["now"]:
        line = []
        for i in ids:
            _, pp = txrx(ser, i, packet(i, INST_READ, struct.pack("<HH", 132, 4)), 4, timeout=0.1, tries=2)
            if not pp or len(pp) != 4:
                continue
            raw = struct.unpack("<i", pp)[0]
            if raw == 0:
                continue          # a desynced read, not a real -180 deg
            deg = (raw - 2048) * 360.0 / 4096.0
            n_ok[i] += 1
            lo[i] = min(lo[i], deg)
            hi[i] = max(hi[i], deg)
            nm = names.get(i, str(i))
            line.append(f"{SHORT.get(nm, nm)}{deg:+6.1f}[{lo[i]:+6.1f},{hi[i]:+6.1f}]")
        print(" " + " ".join(line) + "   ", end="\r", flush=True)
        # 5 Hz. A tight poll desyncs the bus and starts returning zeros, and
        # nobody moves a joint to its stop faster than this anyway.
        time.sleep(0.2)
    print("\n\nmeasured travel:")
    bad = []
    for i in ids:
        if lo[i] > 9e8:
            # Silently dropping these is how a re-sweep of the right knee came
            # back with the right knee simply absent from the summary.
            print(f"  {names.get(i, i):16s} NO DATA -- the servo never answered during the sweep")
            bad.append(names.get(i, i))
            continue
        home = HOME_DEG.get(i)
        note = ""
        if home is not None and not (lo[i] - 2 <= home <= hi[i] + 2):
            note = f"   <-- INCOMPLETE: never reached home ({home:+.1f})"
            bad.append(names.get(i, i))
        span = hi[i] - lo[i]
        if span < 5.0 and not note:
            note = f"   <-- barely moved ({span:.1f} deg swept)"
            bad.append(names.get(i, i))
        print(f"  {names.get(i, i):16s} [{lo[i]:+7.1f}, {hi[i]:+7.1f}] deg   "
              f"= [{lo[i] * 3.14159 / 180:+.4f}, {hi[i] * 3.14159 / 180:+.4f}] rad"
              f"  {n_ok[i]:4d} samples{note}")
    # Mirror cross-check: left [a,b] should predict right [-b,-a].
    done = set()
    for i in ids:
        j = MIRROR.get(i)
        if j is None or j not in ids or lo[i] > 9e8 or lo[j] > 9e8 or (j, i) in done:
            continue
        done.add((i, j))
        pred_lo, pred_hi = -hi[i], -lo[i]
        err = max(abs(pred_lo - lo[j]), abs(pred_hi - hi[j]))
        verdict = "consistent" if err < 8 else f"DISAGREE by {err:.0f} deg -- one side under-swept"
        print(f"  mirror {names.get(i, i)} vs {names.get(j, j)}: "
              f"predicts [{pred_lo:+7.1f},{pred_hi:+7.1f}] measured "
              f"[{lo[j]:+7.1f},{hi[j]:+7.1f}]  {verdict}")
    if bad:
        print(f"\n  re-sweep: {', '.join(map(str, bad))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyS2")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--ids", default="10-14,20-24,30-34",
                    help="the real map, from duck-control model.rs JOINT_IDS: "
                         "right leg 10-14, left leg 20-24, neck/head/mouth 30-34")
    ap.add_argument("--limits", action="store_true",
                    help="also read Min/Max Position Limit and Present Position. The XL330 "
                         "refuses a goal outside its position limits and will sit against a "
                         "mechanical stop drawing current, which is how an ankle overloads.")
    ap.add_argument("--watch", metavar="NAMES", nargs="?", const=PLACEHOLDER_JOINTS, default=None,
                    help="live min/max of these joints (comma-separated names or ids) with "
                         "torque OFF, so you can move each one to its mechanical stops and "
                         "read the real travel. The MJCF gives hip_pitch, knee and ankle an "
                         "exported placeholder of exactly +/-90 deg; measuring them is the "
                         "point of this mode.")
    ap.add_argument("--reboot", action="store_true",
                    help="send REBOOT to every id whose error register is non-zero. "
                         "This CLEARS the latch; it does not fix a real fault.")
    a = ap.parse_args()

    ids = []
    for part in a.ids.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            ids += list(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    ser = _Port(a.port, a.baud)
    if a.watch:
        by_name = {v: k for k, v in NAMES.items()}
        want = [by_name.get(w.strip(), None) or (int(w) if w.strip().isdigit() else None)
                for w in a.watch.split(",")]
        want = [w for w in want if w is not None]
        return _watch(ser, want, NAMES)
    faulted, seen, fault_bits = [], 0, {}
    for i in ids:
        err, payload = txrx(ser, i, packet(i, INST_READ, struct.pack("<HH", ADDR_HW_ERROR, 1)), 1)
        if payload is None or not payload:
            continue
        seen += 1
        hw = payload[0]
        _, v = txrx(ser, i, packet(i, INST_READ, struct.pack("<HH", ADDR_VOLT, 2)), 2)
        _, t = txrx(ser, i, packet(i, INST_READ, struct.pack("<HH", ADDR_TEMP, 1)), 1)
        volts = struct.unpack("<H", v)[0] / 10.0 if v and len(v) == 2 else float("nan")
        temp = t[0] if t else -1
        flag = "" if hw == 0 else "  <-- " + ", ".join(
            name for bit, name in BITS.items() if hw & (1 << bit)) or f"  <-- unknown bits 0x{hw:02x}"
        extra = ""
        if a.limits:
            _, mx = txrx(ser, i, packet(i, INST_READ, struct.pack("<HH", 48, 4)), 4)
            _, mn = txrx(ser, i, packet(i, INST_READ, struct.pack("<HH", 52, 4)), 4)
            _, pp = txrx(ser, i, packet(i, INST_READ, struct.pack("<HH", 132, 4)), 4)
            def deg(b, signed=False):
                if not b or len(b) != 4:
                    return float("nan")
                v = struct.unpack("<i" if signed else "<I", b)[0]
                return (v - 2048) * 360.0 / 4096.0      # 0.088 deg/tick, 2048 = centre
            extra = (f"  limits [{deg(mn):+7.1f},{deg(mx):+7.1f}] deg"
                     f"  at {deg(pp, True):+7.1f}")
        latched = "  TORQUE LATCHED OFF" if hw & LATCHING else ""
        print(f"  id {i:3d} {NAMES.get(i, '?'):16s} hw_error=0x{hw:02x}  "
              f"{volts:5.1f} V  {temp:3d} C{extra}{flag if not a.limits else ''}{latched}")
        if hw:
            faulted.append(i)
            fault_bits[i] = hw

    hard = [i for i in faulted if fault_bits[i] & LATCHING]
    print(f"\n{seen} devices answered. {len(faulted)} flagged; "
          f"{len(hard)} with a TORQUE-LATCHING fault: "
          f"{[f'{i} {NAMES.get(i, chr(63))}' for i in hard] or 'none'}")
    if faulted and not hard:
        print("  (all flags are bit 0 input-voltage, which does not latch torque -- "
              "chronic on this robot's 2S pack and not a fault to chase)")
    if faulted and a.reboot:
        for i in faulted:
            ser.write(packet(i, INST_REBOOT))
            ser.flush()
            time.sleep(0.05)
        print(f"rebooted {faulted} — latch cleared. Restart robotd and re-check.")
    elif faulted:
        print("re-run with --reboot to clear the latch (does not fix a real fault).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
