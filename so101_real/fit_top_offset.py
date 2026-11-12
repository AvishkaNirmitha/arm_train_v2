"""Measure the top camera -> arm offset from recorded picks (for snap_top.py / the click tool's M = snap).

For every recorded pick: where the top camera sees the block (snap_top's box fit on the picture before your first
top click) vs. where the gripper really was when it closed on it (holding: the gripper stopped above 41). The mean
difference is the offset; the leave-one-out error says how well a snapped click + this offset would have done.

    .venv/bin/python so101_real/fit_top_offset.py recordings/11_11_2026_pick_white_cube_3 recordings/12_11_2026_pick_white_cube_4

Needs build_decisions.py run on those folders first. Writes so101_real/top_arm_offset.json. Run it again after
recalibrating the top camera (align_top_camera.py) or the arm (joint map).
"""
import os
import sys
import json
import bisect
import argparse
from datetime import datetime

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import click_goal_real as C                                           # noqa: E402
from real_arm_client import load_joint_map, real_to_sim               # noqa: E402
from snap_top import TopCam, snap, BLOCK_TOP, OFFSET_FILE              # noqa: E402

HOLDING = 41.0                   # gripper reading above this after a close = it holds the block


def close_tcp(ep, kin, env, jm):
    """(gripper point when it first closed on the block, time) or None."""
    E = [json.loads(line) for line in open(os.path.join(ep, "events.jsonl")) if line.strip()]
    F = [json.loads(line) for line in open(os.path.join(ep, "frames.jsonl")) if line.strip()]
    cl = [e for e in E if e["kind"] == "gripper_close" and e.get("source") != "episode start"
          and e.get("moved") is not False and (e.get("now") or 0) > HOLDING]
    if not cl:
        return None
    st = [e for e in E if e["kind"] == "move_start" and e["t"] <= cl[0]["t"] and set(e.get("target") or {}) == {"gripper"}]
    t = st[-1]["t"] if st else cl[0]["t"]
    i = max(bisect.bisect_left([f["t"] for f in F], t) - 1, 0)
    return kin.set(real_to_sim(F[i]["state"], jm, env.q_home)).tcp_pos(), cl[0]["t"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recordings", nargs="+", help="recording folders (with decisions/ from build_decisions.py)")
    ap.add_argument("--dry", action="store_true", help="only print, don't write top_arm_offset.json")
    args = ap.parse_args()

    env = C.PickEnv(randomize=False)
    kin = C.Kin(env)
    jm = load_joint_map()
    with open(C.TOP_CAM_CALIB) as f:
        calib = json.load(f)
    TW, TH = calib["image_size"]
    cam_full = TopCam.from_mujoco(kin.m, kin.set(env.q_home).d, TW, TH)
    poly_full = np.array([calib["corners_px"][c] for c in C.align.CORNERS], float)

    seen, arm, names = [], [], []
    for root in args.recordings:
        path = os.path.join(root, "decisions", "decisions.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found - run build_decisions.py {root} first")
        rows = [json.loads(line) for line in open(path) if line.strip()]
        for ep in sorted({r["episode"] for r in rows}):
            c = close_tcp(os.path.join(root, ep), kin, env, jm)
            if c is None:
                continue
            tcp, t_close = c
            first = [r for r in rows if r["episode"] == ep and r["label"] == "click_top" and r["t"] < t_close]
            if not first:
                continue
            r = first[0]
            img = cv2.imread(os.path.join(root, "decisions", r["images"]["top"]))
            h, w = img.shape[:2]
            s = w / TW
            out = snap(img, r["point"]["u"] * s, r["point"]["v"] * s, cam_full.resized(w, h), poly_full * s)
            name = f"{os.path.basename(root.rstrip('/'))}/{ep}"
            if out["kind"] != "block":
                print(f"  {name}: skipped ({out.get('why') or 'snapped to the ' + str(out['kind'])})")
                continue
            seen.append(out["point"][:2])
            arm.append(tcp[:2])
            names.append(name)
    if len(seen) < 5:
        raise SystemExit(f"only {len(seen)} usable picks - need at least 5")
    S, G = np.array(seen), np.array(arm)
    d = G - S
    off = d.mean(0)
    loo = [np.linalg.norm(S[i] + np.delete(d, i, 0).mean(0) - G[i]) * 1000 for i in range(len(S))]
    raw = [np.linalg.norm(G[i] - S[i]) * 1000 for i in range(len(S))]
    for n, dd, e in zip(names, d * 1000, loo):
        print(f"  {n:<48} arm needed {dd[0]:+5.1f} {dd[1]:+5.1f} mm   (with the mean offset: {e:4.1f} mm off)")
    print(f"\n{len(S)} picks. offset x {off[0] * 1000:+.1f} y {off[1] * 1000:+.1f} mm (sd {d[:, 0].std() * 1000:.1f} / "
          f"{d[:, 1].std() * 1000:.1f}). Snapped click without it: median {np.median(raw):.1f} mm off, with it "
          f"(leave-one-out): median {np.median(loo):.1f}, max {np.max(loo):.1f} mm")
    if not args.dry:
        with open(OFFSET_FILE, "w") as f:
            json.dump({"offset_m": [round(float(off[0]), 5), round(float(off[1]), 5)],
                       "sd_mm": [round(float(d[:, 0].std() * 1000), 1), round(float(d[:, 1].std() * 1000), 1)],
                       "loo_median_mm": round(float(np.median(loo)), 1), "picks": len(S),
                       "recordings": args.recordings, "top_calib_created": calib.get("created"),
                       "created": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
        print("written:", OFFSET_FILE)


if __name__ == "__main__":
    main()
