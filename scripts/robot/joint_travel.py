#!/usr/bin/env python3
"""Record joint travel. Runs ON the robot, with torque off.

    sudo systemctl stop robotd
    robotctl robot relax          # (before stopping robotd, or power was already off)
    sudo python3 joint_travel.py

Move each joint to both of its stops. Ctrl-C prints the measured range.

ONE SYNC READ PER CYCLE, not one round trip per joint. The per-joint version of
this went through three field attempts and produced a different answer each
time -- dropouts reported as silence, joints missing from the summary, and a
tcflush before every write that discarded replies already in flight. A single
Sync Read (instruction 0x82) asks every servo at once and each answers in turn,
which is what the daemon itself does, and it removes the per-joint retry logic
that was the source of the mess.
"""
import argparse
import os
import select
import struct
import sys
import termios
import time

ADDR_PRESENT_POSITION, LEN_POSITION = 132, 4
INST_SYNC_READ = 0x82
BROADCAST = 0xFE

# duck-control model.rs: JOINT_IDS and DEFAULT_POSITION (degrees).
JOINTS = [
    (20, "left_hip_yaw", 0.0), (21, "left_hip_roll", -5.0), (22, "left_hip_pitch", -26.2),
    (23, "left_knee", -0.3), (24, "left_ankle", +26.0),
    (30, "neck_pitch", +20.0), (31, "head_pitch", +20.0), (32, "head_yaw", 0.0),
    (33, "head_roll", 0.0), (34, "mouth", 0.0),
    (10, "right_hip_yaw", 0.0), (11, "right_hip_roll", +5.0), (12, "right_hip_pitch", +26.2),
    (13, "right_knee", +0.3), (14, "right_ankle", -26.0),
]
NAME = {i: n for i, n, _ in JOINTS}
HOME = {i: h for i, _, h in JOINTS}
MIRROR = {20: 10, 21: 11, 22: 12, 23: 13, 24: 14,
          10: 20, 11: 21, 12: 22, 13: 23, 14: 24}
DEFAULT = "22,23,24,12,13,14"       # hip_pitch, knee, ankle -- the MJCF placeholders


def crc16(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def build(dxl_id: int, inst: int, params: bytes) -> bytes:
    body = (bytes([0xFF, 0xFF, 0xFD, 0x00, dxl_id])
            + struct.pack("<H", len(params) + 3) + bytes([inst]) + params)
    return body + struct.pack("<H", crc16(body))


def sync_read_packet(ids) -> bytes:
    params = struct.pack("<HH", ADDR_PRESENT_POSITION, LEN_POSITION) + bytes(ids)
    return build(BROADCAST, INST_SYNC_READ, params)


def frames(buf: bytes):
    """Yield (id, err, params) for every CRC-valid STATUS packet in buf."""
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
        f = buf[i:end]
        if f[7] == 0x55 and crc16(f[:-2]) == struct.unpack("<H", f[-2:])[0]:
            yield f[4], f[8], f[9:-2]
        i += 4


class Port:
    """Blocking-with-timeout serial. No O_NONBLOCK busy loop, no per-write flush."""

    def __init__(self, path: str, baud: int):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
        a = termios.tcgetattr(self.fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = getattr(termios, f"B{baud}")
        a[6] = list(a[6])
        a[6][termios.VMIN] = 0
        a[6][termios.VTIME] = 1
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        termios.tcflush(self.fd, termios.TCIOFLUSH)   # once, at open

    def ask(self, pkt: bytes, want: int, timeout: float):
        # Flush BEFORE the write, not after. Replies that arrived past the last
        # deadline otherwise sit in the kernel buffer and shift every frame
        # boundary in the next parse, which is what turned a working first
        # cycle into a stream of raw-zero reads.
        termios.tcflush(self.fd, termios.TCIFLUSH)
        os.write(self.fd, pkt)
        termios.tcdrain(self.fd)
        buf, got, deadline = b"", {}, time.time() + timeout
        while time.time() < deadline and len(got) < want:
            if select.select([self.fd], [], [], max(0.0, deadline - time.time()))[0]:
                chunk = os.read(self.fd, 256)
                if chunk:
                    buf += chunk
                    for fid, _err, params in frames(buf):
                        if len(params) < LEN_POSITION:
                            continue
                        raw = struct.unpack("<i", params[:LEN_POSITION])[0]
                        # Raw 0 is -180 deg, which no joint on this robot can
                        # reach; it only appears on a desynced read.
                        if raw != 0:
                            got[fid] = raw
        return got


def ticks_to_deg(raw: int) -> float:
    return (raw - 2048) * 360.0 / 4096.0        # bus.rs: (2*pi*raw/4096) - pi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyS2")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--joints", default=DEFAULT, help="ids or names, comma separated")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this long and print the summary, instead of waiting for "
                         "Ctrl-C. Ctrl-C still works; this exists so the sweep can be driven "
                         "from a script without a signal racing the wrapper's cleanup trap.")
    a = ap.parse_args()

    by_name = {n: i for i, n, _ in JOINTS}
    ids = []
    for tok in a.joints.split(","):
        tok = tok.strip()
        ids.append(int(tok) if tok.isdigit() else by_name[tok])

    port = Port(a.port, a.baud)
    pkt = sync_read_packet(ids)
    lo = {i: None for i in ids}
    hi = {i: None for i in ids}
    n = {i: 0 for i in ids}

    print("Torque OFF. Move each joint slowly to BOTH stops. Ctrl-C when done.\n")
    t_end = time.time() + a.seconds if a.seconds else None
    try:
        while t_end is None or time.time() < t_end:
            # Short timeout, and partial replies are fine: min/max is tracked per
            # joint, so a joint that misses a cycle simply has one fewer sample.
            # At 0.25 s a cycle that lost one reply blocked the whole sweep, and
            # a 30 s hand sweep came back with 11 samples -- nowhere near enough
            # to catch a stop you pass through once.
            got = port.ask(pkt, len(ids), timeout=0.04)
            cells = []
            for i in ids:
                if i in got:
                    d = ticks_to_deg(got[i])
                    n[i] += 1
                    lo[i] = d if lo[i] is None else min(lo[i], d)
                    hi[i] = d if hi[i] is None else max(hi[i], d)
                    cells.append(f"{NAME[i][:9]:>9s}{d:+7.1f}")
                else:
                    cells.append(f"{NAME[i][:9]:>9s}   ----")
            print("  " + " ".join(cells), end="\r", flush=True)
            time.sleep(1.0 / a.hz)
    except KeyboardInterrupt:
        pass

    print("\n\nmeasured travel:")
    bad = []
    for i in ids:
        if n[i] == 0:
            print(f"  {NAME[i]:16s} NO DATA — never answered")
            bad.append(NAME[i])
            continue
        span = hi[i] - lo[i]
        note = ""
        if span < 5.0:
            note = f"  <-- barely moved ({span:.1f} deg)"
            bad.append(NAME[i])
        elif not (lo[i] - 2 <= HOME[i] <= hi[i] + 2):
            note = f"  <-- never reached home ({HOME[i]:+.1f})"
            bad.append(NAME[i])
        print(f"  {NAME[i]:16s} [{lo[i]:+7.1f}, {hi[i]:+7.1f}] deg = "
              f"[{lo[i] * 3.14159265 / 180:+.4f}, {hi[i] * 3.14159265 / 180:+.4f}] rad"
              f"  {n[i]:5d} samples{note}")

    seen = set()
    for i in ids:
        j = MIRROR.get(i)
        if j in ids and n[i] and n.get(j) and (j, i) not in seen:
            seen.add((i, j))
            pl, ph = -hi[i], -lo[i]
            err = max(abs(pl - lo[j]), abs(ph - hi[j]))
            print(f"  mirror {NAME[i]} vs {NAME[j]}: predicts [{pl:+7.1f},{ph:+7.1f}], "
                  f"got [{lo[j]:+7.1f},{hi[j]:+7.1f}], max err {err:.1f} deg "
                  f"{'AGREE' if err < 10 else '<-- one side under-swept'}")
    if bad:
        print(f"\n  re-sweep: {', '.join(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
