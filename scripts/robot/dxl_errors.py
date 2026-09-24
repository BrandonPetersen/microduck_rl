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
        self.os.write(self.fd, data)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyS2")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--ids", default="10-14,20-24,30-34",
                    help="the real map, from duck-control model.rs JOINT_IDS: "
                         "right leg 10-14, left leg 20-24, neck/head/mouth 30-34")
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
        latched = "  TORQUE LATCHED OFF" if hw & LATCHING else ""
        print(f"  id {i:3d} {NAMES.get(i, '?'):16s} hw_error=0x{hw:02x}  "
              f"{volts:5.1f} V  {temp:3d} C{flag}{latched}")
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
