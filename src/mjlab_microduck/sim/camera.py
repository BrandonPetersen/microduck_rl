"""What a duck sees, in the format `mediad` captures.

MuJoCo renders RGB; `mediad` pins its pipeline to UYVY because that is what `v4l2src` can drive at
full rate off the rkisp — so the conversion happens here, on the side that knows it is a simulator.

**Frames do not go down the JSON link.** 640x360 UYVY is 460,800 bytes, and at 15 fps that is 6.9
MB/s — JSON would be absurd. So a camera is its own TCP port carrying length-prefixed raw frames:
four bytes of little-endian length, then the bytes, forever. No handshake, because there is nothing
to negotiate that both ends do not already have to agree on to be useful.

**Opt in, per duck.** Rendering is the most expensive thing in the simulator by a wide margin —
12.2 ms per 640x360 frame, measured, against 0.3 ms to step four ducks' physics. Four ducks with
cameras at 15 fps is most of a core; four without is nothing. Most sessions do not need one.
"""

from __future__ import annotations

import socket
import socketserver
import struct
import threading
import time
from collections import deque

import mujoco
import numpy as np

# 16:9 at a size the default offscreen framebuffer can hold — MuJoCo caps offscreen rendering at
# the model's `<global offwidth/offheight>`, which is 640x480 unless the scene says otherwise.
WIDTH = 640
HEIGHT = 360

# The sensor's rate is 30, but a rendered frame costs 12 ms and a duck that is being watched is
# usually being watched rather than raced. 15 halves the cost for something nobody can see.
FPS = 15

# The far plane the SLAM scenes pin (see scene_vslam.xml). MuJoCo returns exactly this value for
# pixels where no geometry was hit, so a consumer must mask them rather than feed them to SLAM.
ZFAR_M = 20.0

# How many rendered frames wait for the bench reader. A queue rather than a single "latest" slot,
# because a ground-truth benchmark must not lose frames to a reader that falls a little behind.
# Bounded, because an unbounded one turns a slow reader into an out-of-memory crash instead; past
# this depth a frame IS dropped, and the drop shows up as a jump in `seq`, so it can never be
# silent. Eight 640x360 frames of UYVY plus float32 depth is about 11 MB.
BENCH_QUEUE = 8

# BT.601, the same coefficients `duck_detect`'s one-pass sampler uses on the robot.
_Y = np.array([0.299, 0.587, 0.114])
_U = np.array([-0.168736, -0.331264, 0.5])
_V = np.array([0.5, -0.418688, -0.081312])


def to_uyvy(rgb: np.ndarray) -> bytes:
    """RGB to packed UYVY: `U Y0 V Y1` per pixel pair, chroma averaged across the pair.

    Averaged rather than dropped, because a subsampler that takes the left pixel's chroma puts a
    half-pixel colour shift into every frame — invisible on a duck and not invisible to a detector
    trained on a real camera.
    """
    frame = rgb.astype(np.float32)
    luma = frame @ _Y
    chroma_u = frame @ _U + 128.0
    chroma_v = frame @ _V + 128.0

    pairs = frame.shape[1] // 2
    packed = np.empty((frame.shape[0], pairs, 4), dtype=np.uint8)
    packed[:, :, 0] = np.clip((chroma_u[:, 0::2] + chroma_u[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 1] = np.clip(luma[:, 0::2], 0, 255)
    packed[:, :, 2] = np.clip((chroma_v[:, 0::2] + chroma_v[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 3] = np.clip(luma[:, 1::2], 0, 255)
    return packed.tobytes()


class Camera:
    """One duck's head camera, rendered on demand.

    The renderer is not thread-safe and is expensive to make, so one lives here and only the frame
    loop touches it.
    """

    def __init__(self, model: mujoco.MjModel, name: str, width: int = WIDTH, height: int = HEIGHT):
        self.camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.camera < 0:
            raise SystemExit(f"the model has no camera {name!r}")

        # The camera's 180-degree mount correction now lives in the MJCF (`quat="0 0 0 1"`, was
        # `"0 0 -1 0"`), not here. It used to be applied by mutating `model.cam_quat` in this
        # constructor, which meant anything reading `data.cam_xmat` without building a Camera --
        # the truth op, a debug script -- saw a camera facing backwards.
        #
        # The rendered frame is still rolled a quarter turn from the FK-published camera frame
        # (`robot.state.frames.camera`), and that is DELIBERATE: the real head camera is mounted a
        # quarter turn off, `mediad --rotate 90` announces it, and every consumer already undoes
        # it. `tests/test_sim_camera_frame.py` pins the relationship.
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        # Depth needs its OWN renderer: `render()` sets mjRND_SEGMENT|mjRND_IDCOLOR for a depth
        # pass, so one instance cannot serve both. Both are driven from a single `update_scene`
        # below, which is what makes the two images the same instant and pixel-aligned.
        self.depth_renderer = mujoco.Renderer(model, height=height, width=width)
        self.depth_renderer.enable_depth_rendering()
        self.width = width
        self.height = height
        self.latest: bytes | None = None
        self.latest_depth: np.ndarray | None = None
        self.sim_time = 0.0
        self.mono_ns = 0
        self.seq = -1
        self.lock = threading.Lock()
        # Frames waiting for the bench reader, oldest first. Waiters block on this instead of
        # polling on their own timer, so every rendered frame is delivered exactly once and in
        # order (see FrameHandler's beat, fixed in BenchHandler).
        self._pending: deque = deque(maxlen=BENCH_QUEUE)
        self.cond = threading.Condition(self.lock)

    def render(self, world) -> None:
        """Render one RGB frame and one ground-truth depth frame, from one scene copy.

        **`update_scene` reads the whole of `MjData`, and it runs on the step loop's thread while
        sensor reads run on socket threads.** Unlocked, a ToF read caught a site orientation
        mid-write and got a zero-length ray direction — which MuJoCo answers with
        `mj_ray: vector length is too small` and an abort, taking the simulator down with it. The
        lock is held for the two scene copies and the clock read, which is a couple of
        milliseconds, and released for the renders, which are ~14 ms combined in `scene_vslam.xml`
        and touch no shared state.

        `sim_time` is read INSIDE the lock, with the scene copy. Read outside it, the timestamp
        would belong to a later world than the pixels — which is the whole failure this bench
        exists to remove.
        """
        with world.lock:
            sim_time = float(world.data.time)
            self.renderer.update_scene(world.data, camera=self.camera)
            self.depth_renderer.update_scene(world.data, camera=self.camera)
        packed = to_uyvy(self.renderer.render())
        # float32 metres, depth along the optical axis; exactly ZFAR_M where nothing was hit.
        depth = np.ascontiguousarray(self.depth_renderer.render(), dtype=np.float32)
        mono_ns = time.monotonic_ns()
        with self.cond:
            self.seq += 1
            self.latest = packed
            self.latest_depth = depth
            self.sim_time = sim_time
            self.mono_ns = mono_ns
            self._pending.append((self.seq, sim_time, mono_ns, packed, depth))
            self.cond.notify_all()

    def frame(self) -> bytes | None:
        """The UYVY frame alone — what `mediad` reads off port 7901. Unchanged on purpose."""
        with self.lock:
            return self.latest

    def wait_frame(self, after_seq: int, timeout: float) -> tuple[int, float, int, bytes, np.ndarray] | None:
        """Block until a frame newer than `after_seq` is queued; return the OLDEST such, or None.

        Returns `(seq, sim_time, mono_ns, uyvy, depth_m)`. Blocking on a bounded QUEUE rather than
        polling a single "latest" slot is what makes the bench stream gapless. Port 7901 has the
        opposite arrangement and both of its failure modes: the render loop runs at 16.67 Hz
        (`passes_per_eye = round((1/15)/0.02) = 3`) while `FrameHandler` re-sends `latest` on its
        own 15 Hz timer, so roughly 1.7 frames a second are duplicated or dropped — invisibly,
        because nothing on that wire carries a sequence number.

        A reader more than `BENCH_QUEUE` frames behind does still lose frames, and that is the
        point of returning `seq`: the loss appears as a gap the consumer can detect rather than as
        silently wrong data.
        """
        with self.cond:
            if not self.cond.wait_for(
                lambda: bool(self._pending) and self._pending[-1][0] > after_seq, timeout=timeout
            ):
                return None
            # Discard anything the caller has already seen, then hand over the oldest it has not.
            while self._pending and self._pending[0][0] <= after_seq:
                self._pending.popleft()
            return self._pending.popleft() if self._pending else None


class FrameHandler(socketserver.BaseRequestHandler):
    """Length-prefixed frames, at the camera's rate, until the reader goes away."""

    def handle(self) -> None:
        camera: Camera = self.server.camera
        fps: int = self.server.fps
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"== camera: a reader connected from {self.client_address}", flush=True)
        period = 1.0 / max(1, fps)
        import time

        next_frame = time.perf_counter()
        try:
            while True:
                frame = camera.frame()
                if frame is not None:
                    self.request.sendall(struct.pack("<I", len(frame)) + frame)
                next_frame += period
                slack = next_frame - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_frame = time.perf_counter()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        print("== camera: the reader went away", flush=True)


class FrameServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


# ── the bench stream ──────────────────────────────────────────────────────────────────────────
#
# A SECOND port, deliberately. Port 7901's format is `<I length> + bytes` and `mediad` parses
# exactly that (`mediad/src/pipeline.rs:969-999`), so it cannot carry a timestamp without breaking
# the daemon. The bench stream is for a ground-truth benchmark that bypasses `mediad` and WebRTC
# altogether, so it can afford a real header -- and it carries the ground-truth depth too.
BENCH_MAGIC = b"DBF1"
# magic, seq u32, sim_time f64, mono_ns u64, width u16, height u16, rgb_len u32, depth_len u32
BENCH_HEADER = "<4sIdQHHII"


def pack_bench_frame(
    seq: int, sim_time: float, mono_ns: int, width: int, height: int,
    uyvy: bytes, depth: np.ndarray,
) -> bytes:
    """One bench frame: header, then UYVY, then float32 depth in metres."""
    payload = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
    head = struct.pack(
        BENCH_HEADER, BENCH_MAGIC, seq, float(sim_time), int(mono_ns),
        width, height, len(uyvy), len(payload),
    )
    return head + uyvy + payload


class BenchHandler(socketserver.BaseRequestHandler):
    """Header-prefixed RGB+depth frames, one per render, until the reader goes away.

    Blocks on `Camera.wait_frame` rather than sending on its own timer. That is the difference
    from `FrameHandler`: the render loop runs at 16.67 Hz (passes_per_eye=3 at a 20 ms period)
    and FrameHandler's timer at 15 Hz, so the two beat and roughly 1.7 frames a second are
    duplicated or dropped -- invisibly, because nothing on that wire has a sequence number.
    """

    def handle(self) -> None:
        camera: Camera = self.server.camera
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"== bench: a reader connected from {self.client_address}", flush=True)
        # Start from whatever has already been rendered, so a session opens on the NEXT frame
        # rather than on a backlog rendered before anyone was listening. MEASURED without this:
        # the first 8 frames of a 40 s capture (BENCH_QUEUE deep) came out of the queue stale
        # while the recorder's `read`/`tof`/`truth` round trips returned the live world, so the
        # sidecar rows led their frame by 480, 420, 380, 320, 280, 220, 160, 100, 40 ms before
        # settling at the steady-state 20 ms (one physics step). A consumer pairing by row index
        # got a pose up to 480 ms wrong for those frames. Discarding a pre-connection backlog
        # costs nothing: nobody asked for it, and `seq` still starts wherever the sim is, so the
        # gaplessness check is on what was delivered and not on what the sim ever rendered.
        #
        # To be exact about the boundary: `ThreadingTCPServer` returns from `accept` before this
        # thread reaches the latch below, so a render landing in that window is also discarded as
        # backlog even though the reader was technically already connected. That is bounded to one
        # frame, once, at the start of a session -- not the "never skips a live frame" the rest of
        # this class can claim.
        with camera.lock:
            last = camera.seq
        try:
            while True:
                got = camera.wait_frame(after_seq=last, timeout=5.0)
                if got is None:
                    continue  # nothing rendered in 5 s; the sim may be paused
                seq, sim_time, mono_ns, uyvy, depth = got
                if uyvy is None or depth is None:
                    continue
                last = seq
                self.request.sendall(
                    pack_bench_frame(
                        seq, sim_time, mono_ns, camera.width, camera.height, uyvy, depth
                    )
                )
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        print("== bench: the reader went away", flush=True)


class BenchServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
