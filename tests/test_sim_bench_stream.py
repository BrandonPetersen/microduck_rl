"""The bench frame stream: every rendered frame delivered exactly once, with its own clock.

Port 7901 re-sends whatever `Camera.latest` happens to be, on its own 15 Hz timer, while the
render loop runs at 16.67 Hz -- so frames duplicate and drop at the beat frequency, and none of
them carries a timestamp at all (mediad's AppSrc uses do_timestamp(true), i.e. socket-read time).
The bench stream fixes both: it blocks for a NEW frame, and it carries (seq, sim_time, mono_ns).
"""

import socket
import struct
import threading
import time

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
    cam_pos = [1.0, 2.0, 3.0]
    cam_mat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    blob = pack_bench_frame(7, 1.25, 99, W, H, uyvy, depth, cam_pos, cam_mat)
    size = struct.calcsize(BENCH_HEADER)
    magic, seq, sim_time, mono_ns, w, h, rgb_len, depth_len, *pose = struct.unpack(
        BENCH_HEADER, blob[:size]
    )
    assert magic == BENCH_MAGIC
    assert (seq, sim_time, mono_ns, w, h) == (7, 1.25, 99, W, H)
    assert rgb_len == W * H * 2
    assert depth_len == W * H * 4
    assert pose[:3] == pytest.approx(cam_pos)
    assert pose[3:] == pytest.approx(cam_mat)
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


def test_a_reader_gets_the_next_frame_not_a_pre_connection_backlog():
    """Frames rendered before anyone connected are stale, and the bench recorder pairs each frame
    with a LIVE state/ToF/truth sample -- so handing over a backlog silently mispairs the opening
    frames. MEASURED on a real 40 s capture before this: the first 8 frames' sidecar rows led their
    frame by 480, 420, 380, 320, 280, 220, 160, 100, 40 ms (BENCH_QUEUE deep) instead of the
    steady-state 20 ms, which is one physics step."""
    cam, world = _cam()
    for i in range(5):  # rendered with nobody listening
        world.data.time = 0.1 * i
        cam.render(world)
    server = BenchServer(("127.0.0.1", 0), BenchHandler)
    server.camera = cam
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        sock = socket.create_connection(server.server_address, timeout=5)
        sock.settimeout(5)
        # The handler latches `camera.seq` as its first act, on its own thread. Give that thread
        # time to get there before rendering anything fresh -- rendering has to stay on THIS
        # thread, because a mujoco.Renderer's GL context belongs to the thread that made it.
        time.sleep(0.5)
        world.data.time = 9.0
        cam.render(world)
        size = struct.calcsize(BENCH_HEADER)
        _m, seq, sim_time, _mono, _w, _h, rgb_len, depth_len, *_pose = struct.unpack(
            BENCH_HEADER, _read_exactly(sock, size)
        )
        _read_exactly(sock, rgb_len + depth_len)
        assert (seq, sim_time) == (5, pytest.approx(9.0)), (
            f"got seq {seq} at sim_time {sim_time}, wanted the frame rendered after connect: "
            "the reader was handed the pre-connection backlog"
        )
        sock.close()
    finally:
        server.shutdown()
        server.server_close()


def test_every_render_is_delivered_exactly_once_in_order():
    cam, world = _cam()
    server = BenchServer(("127.0.0.1", 0), BenchHandler)
    server.camera = cam
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
            _m, seq, sim_time, _mono, _w, _h, rgb_len, depth_len, *_pose = struct.unpack(
                BENCH_HEADER, head
            )
            _read_exactly(sock, rgb_len + depth_len)
            seqs.append(seq)
            stamps.append(sim_time)
        assert seqs == list(range(n)), f"duplicated or skipped frames: {seqs}"
        assert stamps == pytest.approx([0.1 * i for i in range(n)])
        sock.close()
    finally:
        server.shutdown()
        server.server_close()


def test_delivered_pose_matches_cam_xpos_xmat_at_the_instant_of_render():
    """The pose in the header must be THIS render's `data.cam_xpos`/`cam_xmat`, not one captured
    outside the lock (where the step loop could have moved on) or reused from a previous frame.
    Proven by moving the camera between two renders and checking the delivered pose moves with
    it, matching each render's own snapshot of `cam_xpos`/`cam_xmat` exactly."""
    cam, world = _cam()
    server = BenchServer(("127.0.0.1", 0), BenchHandler)
    server.camera = cam
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        sock = socket.create_connection(server.server_address, timeout=5)
        sock.settimeout(5)
        size = struct.calcsize(BENCH_HEADER)

        # First render, camera at its model-defined pose.
        world.data.time = 1.0
        mujoco.mj_forward(world.model, world.data)
        expect_pos_1 = world.data.cam_xpos[cam.camera].copy()
        expect_mat_1 = world.data.cam_xmat[cam.camera].reshape(9).copy()
        cam.render(world)

        # Move the camera (via a free joint would need a model change; instead move the whole
        # world's mocap-free scene isn't available here, so perturb cam_xpos/cam_xmat directly --
        # exactly what `render` reads, so a test that only exercises `render`'s own snapshot logic
        # is still meaningful: it proves render captures whatever cam_xpos/cam_xmat says NOW, not
        # a stale copy from before.
        world.data.time = 2.0
        world.data.cam_xpos[cam.camera] = expect_pos_1 + np.array([1.0, 2.0, 3.0])
        world.data.cam_xmat[cam.camera] = np.array(
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )
        expect_pos_2 = world.data.cam_xpos[cam.camera].copy()
        expect_mat_2 = world.data.cam_xmat[cam.camera].reshape(9).copy()
        cam.render(world)

        got = []
        for _ in range(2):
            head = _read_exactly(sock, size)
            _m, _seq, _sim_time, _mono, _w, _h, rgb_len, depth_len, *pose = struct.unpack(
                BENCH_HEADER, head
            )
            _read_exactly(sock, rgb_len + depth_len)
            got.append(pose)
        sock.close()

        np.testing.assert_allclose(got[0][:3], expect_pos_1, atol=1e-12)
        np.testing.assert_allclose(got[0][3:], expect_mat_1, atol=1e-12)
        np.testing.assert_allclose(got[1][:3], expect_pos_2, atol=1e-12)
        np.testing.assert_allclose(got[1][3:], expect_mat_2, atol=1e-12)
        # And the two frames' poses genuinely differ -- otherwise this test would pass even if
        # `render` captured the pose once outside the lock and reused it for every frame.
        assert not np.allclose(got[0], got[1])
    finally:
        server.shutdown()
        server.server_close()
