"""The bench frame stream: every rendered frame delivered exactly once, with its own clock.

Port 7901 re-sends whatever `Camera.latest` happens to be, on its own 15 Hz timer, while the
render loop runs at 16.67 Hz -- so frames duplicate and drop at the beat frequency, and none of
them carries a timestamp at all (mediad's AppSrc uses do_timestamp(true), i.e. socket-read time).
The bench stream fixes both: it blocks for a NEW frame, and it carries (seq, sim_time, mono_ns).
"""

import socket
import struct
import threading

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.camera import (
    BENCH_HEADER,
    BENCH_MAGIC,
    BenchHandler,
    BenchServer,
    Camera,
    pack_bench_frame,
)

W, H = 32, 32


class _World:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.lock = threading.Lock()
        mujoco.mj_forward(model, self.data)


def _cam():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><statistic extent="1.0"/><visual><map znear="0.02" zfar="20"/></visual>'
        '<worldbody><camera name="head_camera" pos="0 0 0" quat="1 0 0 0"/>'
        '<geom type="box" size="5 5 0.01" pos="0 0 -2"/></worldbody></mujoco>'
    )
    world = _World(model)
    return Camera(model, "head_camera", width=W, height=H), world


def test_pack_round_trips_through_the_header():
    uyvy = bytes(W * H * 2)
    depth = np.full((H, W), 1.5, np.float32)
    blob = pack_bench_frame(7, 1.25, 99, W, H, uyvy, depth)
    size = struct.calcsize(BENCH_HEADER)
    magic, seq, sim_time, mono_ns, w, h, rgb_len, depth_len = struct.unpack(
        BENCH_HEADER, blob[:size]
    )
    assert magic == BENCH_MAGIC
    assert (seq, sim_time, mono_ns, w, h) == (7, 1.25, 99, W, H)
    assert rgb_len == W * H * 2
    assert depth_len == W * H * 4
    assert len(blob) == size + rgb_len + depth_len
    back = np.frombuffer(blob[size + rgb_len :], np.float32).reshape(H, W)
    np.testing.assert_allclose(back, depth)


def _read_exactly(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("stream ended early")
        buf += chunk
    return buf


def test_every_render_is_delivered_exactly_once_in_order():
    cam, world = _cam()
    server = BenchServer(("127.0.0.1", 0), BenchHandler)
    server.camera = cam
    server.fps = 15
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        sock = socket.create_connection(server.server_address, timeout=5)
        sock.settimeout(5)
        n = 6
        for i in range(n):
            world.data.time = 0.1 * i
            cam.render(world)
        size = struct.calcsize(BENCH_HEADER)
        seqs, stamps = [], []
        for _ in range(n):
            head = _read_exactly(sock, size)
            _m, seq, sim_time, _mono, _w, _h, rgb_len, depth_len = struct.unpack(BENCH_HEADER, head)
            _read_exactly(sock, rgb_len + depth_len)
            seqs.append(seq)
            stamps.append(sim_time)
        assert seqs == list(range(n)), f"duplicated or skipped frames: {seqs}"
        assert stamps == pytest.approx([0.1 * i for i in range(n)])
        sock.close()
    finally:
        server.shutdown()
        server.server_close()
