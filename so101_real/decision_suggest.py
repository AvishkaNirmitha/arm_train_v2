"""Live answers of the decision model for click_goal_real.py --suggest (suggest mode: you accept with A or do your own).

The model gets what it got in training (build_decisions.py -> decision_dataset.py): the three clean camera pictures
(window size, then 448x336), the gripper reading and the last 3 decisions. The decisions are rebuilt here from the
same events the click tool logs, with the same rules as build_decisions.episode_decisions:
  * a click counts once the move it was for is executed (the last click per camera, one decision per camera)
  * a gripper open / close counts when it moved (not the one at episode start), with where it stopped
  * episode_start (R) forgets them - training episodes start with no history.
The model runs in its own thread (~0.7 s on the laptop GPU); the window keeps updating meanwhile.
"""
import time
import threading

import cv2
from PIL import Image

from decision_dataset import CAMS, IMG_W, IMG_H, build_prompt, parse_answer

N_RECENT = 3                     # decisions in the prompt (as in training: build_decisions --history 3)


class History:
    """The decisions so far, from the click tool's events. sizes = {camera: (w, h)} of the click pixels."""

    def __init__(self, sizes=None):
        self.sizes = sizes or {}
        self.reset()

    def reset(self):
        self.steps, self.pending, self.goal_cams = [], {}, set()

    def event(self, kind, data):
        t = time.time()
        if kind == "episode_start":
            self.reset()
        elif kind == "click":
            self.pending.setdefault(data["camera"], []).append((t, float(data["u"]), float(data["v"])))
        elif kind == "clear":
            self.pending.clear()
        elif kind == "goal":
            self.goal_cams = set(data.get("clicks") or {})
        elif kind == "execute":
            for cam, clicks in self.pending.items():
                if cam in self.goal_cams and cam in self.sizes:
                    w, h = self.sizes[cam]
                    self.steps.append({"t": clicks[0][0], "label": f"click_{cam}",
                                       "x": clicks[-1][1] / w, "y": clicks[-1][2] / h})
            self.pending.clear()
        elif kind in ("gripper_open", "gripper_close"):
            if data.get("source") == "episode start" or data.get("moved") is False:
                return
            self.steps.append({"t": t, "label": kind, "value": data.get("value"), "stopped_at": data.get("now")})
        else:
            return
        self.steps.sort(key=lambda s: s["t"])

    def recent(self):
        return self.steps[-N_RECENT:]


def model_picture(img, show_scale):
    """A live camera frame -> the picture the model was trained on: the window size (what was recorded), then
    448x336. BGR in, RGB PIL out."""
    if show_scale != 1.0:
        img = cv2.resize(img, None, fx=show_scale, fy=show_scale, interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class Suggester:
    def __init__(self, lora_dir, load_in_4bit=False):
        from decision_policy import DecisionModel          # imports unsloth / torch: only with --suggest
        self.model = DecisionModel(lora_dir, load_in_4bit)
        self.history = History()
        self.cv = threading.Condition()
        self.seq, self.job, self.busy = 0, None, False
        self.result, self.fresh = None, False
        threading.Thread(target=self._run, daemon=True, name="decision-model").start()

    def ask(self, frames, gripper_now, show_scale):
        """New question (frames = {camera: BGR frame}); a newer one replaces one not started yet."""
        imgs = [model_picture(frames[c], show_scale[c]) for c in CAMS]
        recent = self.history.recent()
        prompt = build_prompt(gripper_now, recent)
        with self.cv:
            self.seq += 1
            self.job = (self.seq, imgs, prompt, gripper_now, [dict(s) for s in recent])
            self.result = None
            self.cv.notify()

    def thinking(self):
        return self.busy or self.job is not None

    def poll(self):
        """The newest answer once, when it has just arrived (else None)."""
        with self.cv:
            if self.fresh:
                self.fresh = False
                return self.result
        return None

    def take(self):
        """The current answer (and forget it, so it can't be accepted twice)."""
        with self.cv:
            r, self.result, self.fresh = self.result, None, False
            return r

    def _run(self):
        while True:
            with self.cv:
                while self.job is None:
                    self.cv.wait()
                seq, imgs, prompt, grip, recent = self.job
                self.job, self.busy = None, True
            t0 = time.time()
            try:
                text = self.model(imgs, prompt)
            except Exception as ex:                      # keep the tool running
                text = f"error: {ex!r}"
            p = parse_answer(text)
            res = {"text": text, "label": p[0] if p else None, "x": p[1] if p else None, "y": p[2] if p else None,
                   "seconds": round(time.time() - t0, 2), "gripper_now": grip, "recent": recent}
            with self.cv:
                self.busy = False
                if seq == self.seq:                      # no newer question meanwhile
                    self.result, self.fresh = res, True
