"""Episode recording for click_goal_real.py --record: what you saw in the camera windows + the arm + what you did.

    recordings/<dd_mm_yyyy>_<name>/          one folder per day and name (more sessions add more episodes)
      info.json                             what the files are, cameras, resolutions
      episode_000/
        top.mp4  wrist.mp4  base.mp4        the camera windows as shown (overlays included), REC_FPS
        frames.jsonl                        one line per video frame (same index in every video):
                                              {"i", "t" (s from the episode start), "state" (real joints, deg;
                                               gripper 0..100), "action" (what the arm was told: the goal it was
                                               moving to, interpolated during a move), "moving"}
        events.jsonl                        everything you did: clicks, goals, plans, moves + results, gripper, ...
        meta.json                           start / end, frame count, camera indexes, settings
        calib/                              copies of the calibration files in use (top / wrist / base / joint map)

During a move the windows don't update (the move blocks), but the recording goes on in a thread: frames from the
camera grabbers, the arm drawn where it is expected to be; afterwards "state" / "action" of those frames are replaced
by the joint trace arm_server recorded during the move (if it is new enough to send one).
"""
import os
import re
import json
import shutil
import threading
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD_ROOT = os.path.join(os.path.dirname(HERE), "recordings")


class Grabber(threading.Thread):
    """Reads one camera all the time; get() = the newest frame (no stale buffered frames, and frames keep coming
    while the main thread is busy, e.g. during a blocking arm move)."""

    def __init__(self, cap, prep=None, name="cam"):
        super().__init__(daemon=True, name=f"grab-{name}")
        self.cap, self.prep = cap, prep
        self.lock = threading.Lock()
        self.img, self.t = None, 0.0
        self.running = True
        self.start()

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self.prep is not None:
                f = self.prep(f)
            with self.lock:
                self.img, self.t = f, time.time()

    def get(self, timeout=5.0):
        t_end = time.time() + timeout
        while True:
            with self.lock:
                if self.img is not None:
                    return self.img
            if time.time() > t_end:
                raise SystemExit(f"{self.name}: no camera frame for {timeout:.0f} s")
            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.join(timeout=1.0)


def _js(x):
    """JSON-friendly copy: numpy -> lists / floats, floats rounded."""
    if isinstance(x, dict):
        return {str(k): _js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_js(v) for v in x]
    if isinstance(x, np.ndarray):
        return _js(x.tolist())
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _interp(trace, key, t):
    """{motor: value} of trace[*][key] at time t (linear, clamped to the trace)."""
    ts = [row["t"] for row in trace]
    out = {}
    for m in trace[0][key]:
        out[m] = float(np.interp(t, ts, [row[key][m] for row in trace]))
    return out


class Recorder:
    def __init__(self, name, info, calib_files=(), fps=10.0):
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "session"
        self.folder = os.path.join(RECORD_ROOT, datetime.now().strftime("%d_%m_%Y") + "_" + safe)
        os.makedirs(self.folder, exist_ok=True)
        self.fps, self.calib_files, self.info = float(fps), list(calib_files), info
        path = os.path.join(self.folder, "info.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(_js({"name": name, "created": datetime.now().isoformat(timespec="seconds"), "fps": fps,
                               "files": __doc__.split("    recordings/")[1].split("\n\nDuring")[0], **info}), f, indent=2)
        self.lock = threading.Lock()
        self.ep = None

    @property
    def active(self):
        return self.ep is not None

    def elapsed(self):
        return time.time() - self.ep["t0"] if self.ep else 0.0

    def next_index(self):
        nums = [int(m.group(1)) for d in os.listdir(self.folder) if (m := re.fullmatch(r"episode_(\d+)", d))]
        return max(nums, default=-1) + 1

    def start(self):
        import cv2
        idx = self.next_index()
        d = os.path.join(self.folder, f"episode_{idx:03d}")
        os.makedirs(os.path.join(d, "calib"))
        for p in self.calib_files:
            if os.path.exists(p):
                shutil.copy(p, os.path.join(d, "calib"))
        with self.lock:
            self.ep = {"idx": idx, "dir": d, "t0": time.time(), "writers": {}, "sizes": {}, "rows": [], "events": [],
                       "next_t": 0.0, "cv2": cv2, "started": datetime.now().isoformat(timespec="seconds")}
        return idx, d

    def due(self):
        return self.ep is not None and time.time() >= self.ep["next_t"]

    def add(self, views, state, action, moving=False):
        """One frame of every view (if it is time for one, REC_FPS). views: {camera: BGR image}."""
        with self.lock:
            ep = self.ep
            if ep is None:
                return False
            now = time.time()
            if now < ep["next_t"]:
                return False
            cv2 = ep["cv2"]
            for cam, img in views.items():
                if img is None:
                    continue
                w = ep["writers"].get(cam)
                if w is None:
                    h_, w_ = img.shape[:2]
                    w = cv2.VideoWriter(os.path.join(ep["dir"], f"{cam}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                        self.fps, (w_, h_))
                    ep["writers"][cam], ep["sizes"][cam] = w, (w_, h_)
                if (img.shape[1], img.shape[0]) != ep["sizes"][cam]:
                    img = cv2.resize(img, ep["sizes"][cam])
                w.write(img)
            ep["rows"].append({"i": len(ep["rows"]), "t_abs": now, "state": dict(state), "action": dict(action),
                               "moving": bool(moving)})
            step = 1.0 / self.fps
            ep["next_t"] = ep["next_t"] + step if ep["next_t"] + step > now - step else now + step
            return True

    def event(self, kind, **data):
        with self.lock:
            if self.ep is not None:
                self.ep["events"].append({"t": round(time.time() - self.ep["t0"], 3), "kind": kind, **_js(data)})

    def fill_trace(self, trace):
        """Replace state / action of the frames recorded during a move by arm_server's trace [{t, cmd, joints}]."""
        if not trace or len(trace) < 2:
            return
        with self.lock:
            if self.ep is None:
                return
            t0, t1 = trace[0]["t"], trace[-1]["t"]
            for row in self.ep["rows"]:
                if t0 <= row["t_abs"] <= t1:
                    row["state"].update(_interp(trace, "joints", row["t_abs"]))
                    row["action"].update(_interp(trace, "cmd", row["t_abs"]))
                    row["from_trace"] = True

    def stop(self, keep=True, extra=None):
        """End the episode: keep = write it, else delete it. Returns (folder or None, frame count)."""
        with self.lock:
            ep, self.ep = self.ep, None
        if ep is None:
            return None, 0
        for w in ep["writers"].values():
            w.release()
        n = len(ep["rows"])
        if not keep:
            shutil.rmtree(ep["dir"], ignore_errors=True)
            return None, n
        with open(os.path.join(ep["dir"], "frames.jsonl"), "w") as f:
            for row in ep["rows"]:
                r = {"i": row["i"], "t": round(row["t_abs"] - ep["t0"], 3), "state": row["state"],
                     "action": row["action"], "moving": row["moving"]}
                if row.get("from_trace"):
                    r["from_trace"] = True
                f.write(json.dumps(_js(r)) + "\n")
        with open(os.path.join(ep["dir"], "events.jsonl"), "w") as f:
            for e in ep["events"]:
                f.write(json.dumps(e) + "\n")
        dur = time.time() - ep["t0"]
        meta = {"episode": ep["idx"], "started": ep["started"], "ended": datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(dur, 2), "frames": n, "fps_nominal": self.fps,
                "fps_actual": round(n / dur, 2) if dur > 0 else None, "videos": sorted(ep["writers"]),
                "events": len(ep["events"]), **self.info, **(extra or {})}
        with open(os.path.join(ep["dir"], "meta.json"), "w") as f:
            json.dump(_js(meta), f, indent=2)
        return ep["dir"], n
