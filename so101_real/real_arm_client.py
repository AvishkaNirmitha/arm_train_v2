"""Talk to arm_server.py (running in .venv-lerobot) and convert between sim and real joint angles.

Sim joints are radians in so101_new_calib.xml; the real arm reports lerobot degrees (0 = middle of the
calibrated range). Both use the same zero by design, so the default map is real_deg = degrees(sim_rad).
If mirror.py shows a joint moving the wrong way or offset, fix its sign / offset in real_joint_map.json.
"""
import os
import json
from multiprocessing.connection import Client

import numpy as np

from arm_protocol import ADDRESS, AUTHKEY

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_FILE = os.path.join(HERE, "real_joint_map.json")
SIM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
DISABLED = {"wrist_roll", "gripper"}   # not mounted / broken on the real arm for now: never commanded


def load_joint_map():
    """{joint: {"sign": +-1, "offset_deg": x}}; real_deg = sign * sim_deg + offset_deg."""
    if not os.path.exists(MAP_FILE):
        default = {j: {"sign": 1, "offset_deg": 0.0} for j in SIM_JOINTS}
        with open(MAP_FILE, "w") as f:
            json.dump(default, f, indent=2)
    with open(MAP_FILE) as f:
        return json.load(f)


def sim_to_real(q_sim_rad, joint_map):
    """5 sim joints (rad) -> {motor: real deg} for the motors that are driven."""
    out = {}
    for j, q in zip(SIM_JOINTS, q_sim_rad):
        if j in DISABLED:
            continue
        out[j] = joint_map[j]["sign"] * float(np.degrees(q)) + joint_map[j]["offset_deg"]
    return out


def real_to_sim(real_deg, joint_map, fill=None):
    """{motor: real deg} -> 5 sim joints (rad); joints the arm doesn't report are taken from `fill`."""
    q = np.zeros(len(SIM_JOINTS)) if fill is None else np.array(fill, float)
    for i, j in enumerate(SIM_JOINTS):
        if j in real_deg:
            q[i] = np.radians((real_deg[j] - joint_map[j]["offset_deg"]) * joint_map[j]["sign"])
    return q


class ArmClient:
    def __init__(self, address=ADDRESS, authkey=AUTHKEY):
        try:
            self.conn = Client(address, authkey=authkey)
        except ConnectionRefusedError:
            raise SystemExit("arm_server.py is not running: start it in .venv-lerobot first "
                             "(python so101_real/arm_server.py)")

    def _call(self, **msg):
        self.conn.send(msg)
        reply = self.conn.recv()
        if not reply.get("ok", False) and not reply.get("stopped"):
            raise RuntimeError(f"arm_server: {reply.get('error', reply)}")
        return reply

    def read(self):
        return self._call(cmd="read")

    def move(self, target_deg, speed_deg_s=20.0):
        return self._call(cmd="move", target=target_deg, speed=speed_deg_s)

    def free(self, free=True):
        return self._call(cmd="free", free=free)

    def home(self, speed_deg_s=15.0):
        return self._call(cmd="home", speed=speed_deg_s)

    def close(self):
        self.conn.close()
